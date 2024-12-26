import copy
import time
import torch
import random
from sklearn import metrics
import numpy as np
from tqdm import tqdm
import configargparse
from utils import data_loader
from utils.tools import str2bool, AverageMeter, save_model
from models.make_model import TransferNet
import os
from models import rst
import logging
import json

def get_parser():
    """Get default arguments."""
    parser = configargparse.ArgumentParser(
        description="Transfer learning config parser",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    # general configuration
    parser.add("--config", is_config_file=True, help="config file path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--log_dir', type=str, default='log')
    parser.add_argument('--datasets', type=str, default='office_home',choices=["office_home","office31","visda",
                                                                               "domain_net","digits","image_clef"])
    parser.add_argument('--use_amp', type=str2bool, default=False)

    # network related
    parser.add_argument('--model_name', type=str, default='RN50',choices=["RN50", "VIT-B", "RN101"])

    # data loading related
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--src_domain', type=str, required=True)
    parser.add_argument('--tgt_domain', type=str, required=True)
    parser.add_argument('--gendata_dir', type=str, default='')
    parser.add_argument('--use_img2img', default=False, action='store_true')

    # training related
    parser.add_argument('--l_batch_size', type=int, default=32)
    parser.add_argument('--u_batch_size', type=int, default=32)
    parser.add_argument('--n_epoch', type=int, default=20)
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    parser.add_argument("--n_iter_per_epoch", type=int, default=500, help="Used in Iteration-based training")
    parser.add_argument('--rst_threshold', type=float, default=1e-5)
    parser.add_argument('--baseline', default=False, action='store_true')
    parser.add_argument('--pda', default=False, action='store_true')
    parser.add_argument('--rst', default=False, action='store_true')
    parser.add_argument('--clip', default=False, action='store_true')
    parser.add_argument('--use_dapl', default=False, action='store_true')

    # FixMatch
    parser.add_argument('--fixmatch', default=False, action='store_true')
    parser.add_argument('--fixmatch_threshold', type=float, default=0.95)
    parser.add_argument('--fixmatch_factor', type=float, default=0.5)
    
    parser.add_argument('--mixup', default=False, action='store_true')
    parser.add_argument('--mixup_prob', type=float, default=0)
    parser.add_argument('--alpha', type=float, default=0)
    
    parser.add_argument('--cutmix', default=False, action='store_true')
    parser.add_argument('--cutmix_prob', type=float, default=0)
    parser.add_argument('--beta', type=float, default=0)

    parser.add_argument('--resizemix', default=False, action='store_true')
    parser.add_argument('--resizemix_prob', type=float, default=0)
    parser.add_argument('--gamma', type=float, default=0)
    parser.add_argument('--lam_max', type=float, default=0)
    parser.add_argument('--lam_min', type=float, default=0)

    # optimizer related
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--multiple_lr_classifier', type=float, default=10)

    # loss related
    parser.add_argument('--lambda1', type=float, default=0.25)
    parser.add_argument('--lambda2', type=float, default=0.1)
    parser.add_argument('--lambda3', type=float, default=0.025)
    parser.add_argument('--clf_loss', type=str, default="cross_entropy")

    # learning rate scheduler related
    parser.add_argument('--scheduler', type=str2bool, default=True)

    # linear scheduler
    parser.add_argument('--lr_gamma', type=float, default=0.0003)
    parser.add_argument('--lr_decay', type=float, default=0.75)

    return parser


def set_random_seed(seed=0):
    # seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(args):
    '''
    src_domain, tgt_domain data to load
    '''
    # Use FixMatch
    use_fixmatch = args.fixmatch
    folder_src = os.path.join(args.data_dir, args.src_domain)
    folder_tgt = os.path.join(args.data_dir, args.tgt_domain)
    folder_gen = args.gendata_dir
    if folder_gen:
        gen_loader, n_class = data_loader.load_data(
            args, folder_gen, 16, infinite_data_loader=True, train=True, weight_sampler=False, num_workers=args.num_workers, folder_src=None)
    else:
        gen_loader, n_class = 0, 0
    
    if hasattr(args, 'folder_gen_flux'):
        gen_loader_flux, n_class = data_loader.load_data(
                args, args.folder_gen_flux, 16, infinite_data_loader=True, train=True, num_workers=args.num_workers)
    else:
        gen_loader_flux, n_class = 0, 0
    
    source_loader, n_class = data_loader.load_data(
        args, folder_src, 16, infinite_data_loader=True, train=True, num_workers=args.num_workers)
    target_train_loader, _ = data_loader.load_data(
        args, folder_tgt, 32, infinite_data_loader=True, train=True, use_fixmatch=use_fixmatch, num_workers=args.num_workers, partial=args.pda)
    target_test_loader, _ = data_loader.load_data(
        args, folder_tgt, 1, infinite_data_loader=False, train=False, num_workers=args.num_workers, partial=args.pda)
    return source_loader, target_train_loader, target_test_loader, gen_loader, gen_loader_flux, n_class

def get_model(args):
    model = TransferNet(args).to(args.device)
    return model

def predict(target_test_loader, model, args):
    model.eval()
    list_r = []
    first_test = True
    with torch.no_grad():
        for data, target in tqdm(iterable=target_test_loader):
            data, target = data.to(args.device), target.to(args.device)
            s_output = model.clip_predict(data)
            list_r.append(s_output)
            pred = torch.max(s_output, 1)[1]
            if first_test:
                all_pred = pred
                all_label = target
                first_test = False
            else:
                all_pred = torch.cat((all_pred, pred), 0)
                all_label = torch.cat((all_label, target), 0)
    result = torch.cat(list_r, dim = 0)
    torch.save(result, f'/kaggle/working/{args.src_domain.lower()[0]}2{args.tgt_domain.lower()[0]}.pt')
    acc = torch.sum(torch.squeeze(all_pred).float() == all_label) / float(all_label.size()[0]) * 100

    print('CLIP: test_acc: {:.4f}'.format(acc))
    


def main():
    parser = get_parser()
    args = parser.parse_args()
    set_random_seed(args.seed)
    source_loader, target_train_loader, target_test_loader, gendata_loader, gendata_loader_flux, num_class = load_data(args)
    setattr(args, "num_class", num_class)
    setattr(args, "max_iter", 15000)
    setattr(args, "device", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model = get_model(args)
    
    predict(target_test_loader, model, args)

if __name__ == "__main__":
    # no mixing method, just only source, gen, target domain
    main()