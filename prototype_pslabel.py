import os
from utils import data_loader
from utils.calculate_prototype import init_prototype
from tqdm import tqdm
import torch
import torch.nn.functional as F
import random
import numpy as np
import configargparse
from utils.tools import str2bool
from models.make_model import TransferNet

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

    # FixMatch
    parser.add_argument('--fixmatch', default=False, action='store_true')
    parser.add_argument('--fixmatch_threshold', type=float, default=0.95)
    parser.add_argument('--fixmatch_factor', type=float, default=0.5)
    
    parser.add_argument('--mixup', default=False, action='store_true')
    parser.add_argument('--mixup_prob', type=float, default=0)
    parser.add_argument('--alpha', type=float, default=0)
    parser.add_argument('--use_dapl', default=False, action='store_true')
    
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
    folder_tgt = os.path.join(args.data_dir, args.tgt_domain)
    folder_gen = args.gendata_dir
    
    # gen_loader, n_class = data_loader.load_data(
    #     args, folder_gen, 32, infinite_data_loader=True, train=True, weight_sampler=False, num_workers=args.num_workers, folder_src=None)
    
    target_train_loader, _ = data_loader.load_data(
        args, folder_tgt, 32, infinite_data_loader=False, train=True, use_fixmatch=False, num_workers=args.num_workers, partial=args.pda)
    return target_train_loader

# def feat_prototype_distance_module(feat, prototypes, class_numbers):
#     N, D = feat.shape
#     feat_proto_distance = -torch.ones((N, class_numbers)).cuda()
#     for i in range(class_numbers):
#         feat_proto_distance[:, i] = torch.norm(prototypes[i].unsqueeze(0) - feat, 2, dim=1,)
#     return feat_proto_distance

# def main():
#     count_1, count_2 = 0, 0
#     prototypes = model.prototypes
#     # torch.save(prototypes, 'p.pt')
#     normalize_prototypes = torch.nn.functional.normalize(prototypes, p=2, dim=1)
#     target_train_loader, _, _ = load_data(args)
#     first_test = True
#     i = 0
#     for data, target in tqdm(iterable=target_train_loader,desc='Experiment: '):
#         data, target = data.to(args.device), target.to(args.device)
#         feat_data = model.base_network.forward_features(data)
#         # normalize_feat_data = torch.nn.functional.normalize(feat_data, p=2, dim=1)
#         # feat_proto_distance = torch.matmul(normalize_feat_data, normalize_prototypes.t()) # N, 65
#         feat_proto_distance = feat_prototype_distance_module(feat_data, prototypes, class_numbers=65)

#         # proto_temperature = 8
#         # # s_output = model.clip_predict(data)
#         # prob_prototype = F.softmax(-feat_proto_distance * proto_temperature, dim=1)
#         # target_pred_mix = prob_prototype
#         # target_pred_mix = F.log_softmax(prob_prototype, dim=1)
#         # max_pred_2, pred_2 = torch.max(target_pred_mix, dim=1)
#         s_output = model.clip_predict(data)
#         s_logits = F.softmax(s_output, dim=1)
#         feat_logits = F.softmax(-feat_proto_distance*8, dim=1)
#         uu_logits = F.softmax(feat_logits*s_logits, dim=1)
#         if i == 0:
#             print(torch.max(uu_logits, dim=1)[0])
#             print(torch.max(uu_logits, dim=1)[1])
#             print(torch.max(s_logits, dim=1)[0])
#             print(torch.max(s_logits, dim=1)[1])
#         i += 1
#         # m = F.softmax(s_output, dim=1)
#         max_pred_1, pred_1 = torch.max(uu_logits, dim=1)
#         # max_pred_2, pred_2 = torch.max(s_logits, dim=1)
        
#         # u = []
#         # for i in range(32):
#         #     if max_pred_1[i] > max_pred_2[i]:
#         #         u.append(pred_1[i])
#         #         count_1 += 1
#         #     else:
#         #         u.append(pred_2[i])
#         #         count_2 += 1
        
#         # pred = torch.Tensor(u)
#         pred = pred_1
#         pred = pred.to('cpu')
#         target = target.to('cpu')
#         if first_test:
#             all_pred = pred
#             all_label = target
#             first_test = False
#         else:
#             all_pred = torch.cat((all_pred, pred), 0)
#             all_label = torch.cat((all_label, target), 0)
    
#     acc = torch.sum(torch.squeeze(all_pred).float() == all_label) / float(all_label.size()[0]) * 100       
#     print(acc)
#     print(count_1, count_2)

def test():
    model.eval()
    first_test = True
    desc = "Clip Testing..." if args.clip else "Testing..."
    target_train_loader = load_data(args)
    with torch.no_grad():
        for data, target in tqdm(iterable=target_train_loader,desc=desc):
            data, target = data.to(args.device), target.to(args.device)
            s_output = model.clip_predict(data)
            pred = torch.max(s_output, 1)[1]
            if first_test:
                all_pred = pred
                all_label = target
                first_test = False
            else:
                all_pred = torch.cat((all_pred, pred), 0)
                all_label = torch.cat((all_label, target), 0)
    acc = torch.sum(torch.squeeze(all_pred).float() == all_label) / float(all_label.size()[0]) * 100       
    print(acc)
    
parser = get_parser()
args = parser.parse_args()
set_random_seed(args.seed)
setattr(args, "device", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
setattr(args, "num_class", 65)
setattr(args, "max_iter", 10000)
device = "cuda"
model = TransferNet(args).to(device)

# main()
test()