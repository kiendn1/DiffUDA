import torch
import torch.nn as nn
from models.backbone import get_backbone
from models import cmkd
import logging
import torch.nn.functional as F
import copy
import numpy as np

_logger = logging.getLogger(__name__)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
    elif classname.find('BatchNorm') != -1:
        m.bias.requires_grad_(False)
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
       m.eval()

def rand_bbox(img_shape, lam, margin=0., count=None):
    """Standard CutMix bounding-box that generates a random square bbox
    based on lambda value. This implementation includes support for
    enforcing a border margin as percent of bbox dimensions.

    Args:
        img_shape (tuple): Image shape as tuple
        lam (float): Cutmix lambda value
        margin (float): Percentage of bbox dimension to enforce as margin
            (reduce amount of box outside image). Default to 0.
        count (int, optional): Number of bbox to generate. Default to None
    """
    ratio = np.sqrt(1 - lam)
    img_h, img_w = img_shape[-2:]
    cut_h, cut_w = int(img_h * ratio), int(img_w * ratio)
    margin_y, margin_x = int(margin * cut_h), int(margin * cut_w)
    cy = np.random.randint(0 + margin_y, img_h - margin_y, size=count)
    cx = np.random.randint(0 + margin_x, img_w - margin_x, size=count)
    yl = np.clip(cy - cut_h // 2, 0, img_h)
    yh = np.clip(cy + cut_h // 2, 0, img_h)
    xl = np.clip(cx - cut_w // 2, 0, img_w)
    xh = np.clip(cx + cut_w // 2, 0, img_w)
    return yl, yh, xl, xh

def one_hot_encoding(gt, num_classes):
    """Change gt_label to one_hot encoding.

    If the shape has 2 or more
    dimensions, return it without encoding.
    Args:
        gt (Tensor): The gt label with shape (N,) or shape (N, */).
        num_classes (int): The number of classes.
    Return:
        Tensor: One hot gt label.
    """
    if gt.ndim == 1:
        # multi-class classification
        return F.one_hot(gt, num_classes=num_classes)
    else:
        # binary classification
        # example. [[0], [1], [1]]
        # multi-label classification
        # example. [[0, 1, 1], [1, 0, 0], [1, 1, 1]]
        return gt

    
class TransferNet(nn.Module):
    def __init__(self, args, train=True):
        super(TransferNet, self).__init__()
        # define the network
        # get the feature extractor and the pretrained head
        self.args = args
        self.num_class = args.num_class
        self.base_network = get_backbone(args)
        # self.teacher_model = copy.deepcopy(self.base_network)
        # self.teacher_model.eval()

        # define the task head
        self.classifier_layer = nn.Sequential(
            nn.BatchNorm1d(self.base_network.output_num),
            nn.LayerNorm(self.base_network.output_num, eps=1e-6),
            nn.Linear(self.base_network.output_num, self.num_class,bias=False))
        self.classifier_layer.apply(weights_init_classifier)
        # self.prototypes = init_prototype(self.base_network, args, '/home/user/code/DiffUDA/images/stable-diffusion/Art')
        # self.prototypes = torch.load('p.pt')

        if train:
            # define the loss functions
            self.cmkd = cmkd.CMKD(args)
            self.clf_loss = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    def forward(self, args, source_img, gen_img, target_img, source_label, gen_label, target_strong=None, label_set=None, tgt_index=None, preds_target=None, test=False):
        print(source_img.device)
        print(target_img.device)
        print(gen_img.device)
        if test:
            return self.predict(target_img)
        self.base_network.apply(fix_bn)
        source_feat = self.base_network.forward_features(source_img)

        # calculate source classification loss Lclf
        source_logits = self.classifier_layer(source_feat)
        source_clf_loss = self.clf_loss(source_logits, source_label)
        
        if gen_img is not None:
            gen_feat = self.base_network.forward_features(gen_img)

            # calculate source classification loss Lclf
            gen_logits = self.classifier_layer(gen_feat)
            gen_clf_loss = self.clf_loss(gen_logits, gen_label)
            
            clf_loss = source_clf_loss + gen_clf_loss
        else:
            clf_loss = source_clf_loss

        if not self.args.baseline:
            source_logits_clip = self.base_network.forward_head(source_feat, src_image=True)
            target_feat = self.base_network.forward_features(target_img)
            
            # gen_logits_clip = self.base_network.forward_head(gen_feat)

            # calculate calibrated probability alignment loss Lcpa
            if tgt_index is None:
                target_clip_logits = self.base_network.forward_head(target_feat)
            else:
                target_clip_logits = preds_target[tgt_index]
            target_logits = self.classifier_layer(target_feat)
            # calculate calibrated gini impurity loss Lcgi
            transfer_loss, target_pred_mix = self.cmkd(target_logits, target_clip_logits, source_logits_clip, source_label,label_set)

            # calculate calibrated gini impurity loss Lcgi
            # use_contrastive_loss = True
            # if use_contrastive_loss:
            #     transfer_loss, target_pred_mix = self.cmkd(target_logits, target_clip_logits, source_logits_clip, 
            #                                                                  source_label,label_set, self.prototypes, target_feat, 
            #                                                                  use_contrastive_loss=True, gen_feat=gen_feat, gen_label=gen_label, source_feat=source_feat)
            #                                         #    gen_logit_clip=gen_logits_clip, gen_label=gen_label)
            # else:
            #     transfer_loss, target_pred_mix = self.cmkd(target_logits, target_clip_logits, source_logits_clip, source_label,label_set, self.prototypes, target_feat)
            #                                         #    gen_logit_clip=gen_logits_clip, gen_label=gen_label)
            # calculate_mean_vector_by_label(self.prototypes, gen_feat, gen_label, alpha=0.99)
            # calculate_mean_vector_by_output(self.prototypes, target_feat, target_pred_mix.detach(), alpha=0.99)

        else:
            transfer_loss = torch.tensor(0).to(source_label.device)

        if self.args.fixmatch and target_strong is not None:
            target_pred = F.softmax(target_logits, dim=1)
            if label_set is not None:
                compl_label_set = list(set(torch.range(0, 64).tolist()) - set(label_set))
                compl_label_set = [int(item) for item in compl_label_set]
                target_pred[:, compl_label_set] = 0.0
            max_prob, pred_u = torch.max(target_pred, dim=-1)
            target_strong_feature = self.base_network.forward_features(target_strong)
            target_strong = self.classifier_layer(target_strong_feature)
            fixmatch_loss = self.args.fixmatch_factor * (F.cross_entropy(target_strong, pred_u.detach(), reduction='none') *
                                                         max_prob.ge(self.args.fixmatch_threshold).float().detach()).mean()

            target_pred_clip = F.softmax(target_clip_logits,dim=-1)
            if label_set is not None:
                target_pred_clip[:, compl_label_set] = 0.0

            max_prob, pred_u = torch.max(target_pred_clip, dim=-1)
            target_strong = self.base_network.forward_head(target_strong_feature)
            fixmatch_loss += self.args.fixmatch_factor * (
                        F.cross_entropy(target_strong, pred_u.detach(), reduction='none') *
                        max_prob.ge(self.args.fixmatch_threshold).float().detach()).mean()
            transfer_loss += fixmatch_loss

        if self.args.pda:
            clf_loss = 0.5 * clf_loss
            transfer_loss = 0.1 * transfer_loss
            
        if args.cutmix:
            r = np.random.rand(1)
            if args.beta > 0 and r < args.cutmix_prob:
                one_hot_gen_gt_label = one_hot_encoding(gen_label, num_classes=65)
                # generate mixed sample
                lam = np.random.beta(args.beta, args.beta)
                rand_index = torch.randperm(target_img.size()[0])
                mix_img = target_img.clone().detach()
                bby1, bby2, bbx1, bbx2 = rand_bbox(mix_img.size(), lam)
                mix_img[:, :, bby1:bby2, bbx1:bbx2] = gen_img[rand_index, :, bby1:bby2, bbx1:bbx2]
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (gen_img.size()[-1] * gen_img.size()[-2]))
                mixed_gt_label = lam * target_pred_mix + (1 - lam) * one_hot_gen_gt_label[rand_index, :]
                # compute output
                mix_output = self.base_network.forward_features(mix_img)
                mix_logits = self.classifier_layer(mix_output)
                mix_clf_loss = self.clf_loss(mix_logits, mixed_gt_label)
                # mix_clf_loss = self.clf_loss(mix_logits, target_pred_mix) * lam + self.clf_loss(mix_logits, one_hot_gen_gt_label[rand_index, :]) * (1. - lam)
                
                clf_loss = clf_loss + mix_clf_loss

        if args.resizemix:
            r = np.random.rand(1)
            if args.gamma > 0 and r < args.resizemix_prob:
                one_hot_gen_gt_label = one_hot_encoding(gen_label, num_classes=65)
                # one_hot_src_gt_label = one_hot_encoding(source_label, num_classes=65)
                mix_img = target_img.clone().detach()

                lam = np.random.beta(args.gamma, args.gamma)
                lam = lam * (args.lam_max - args.lam_min) + args.lam_min
                batch_size = gen_img.size(0)
                index = torch.randperm(batch_size)

                bby1, bby2, bbx1, bbx2 = rand_bbox(gen_img.shape, lam)
                mix_img[:, :, bby1:bby2, bbx1:bbx2] = F.interpolate(
                    gen_img[index],
                    size=(bby2 - bby1, bbx2 - bbx1),
                    mode='bilinear')
                bbox_area = (bby2 - bby1) * (bbx2 - bbx1)
                lam = 1. - bbox_area / float(gen_img.size()[-1] * gen_img.size()[-2])
                mixed_gt_label = lam * target_pred_mix + (1 - lam) * one_hot_gen_gt_label[index, :]

                mix_output = self.base_network.forward_features(mix_img)
                mix_logits = self.classifier_layer(mix_output)
                mix_clf_loss = self.clf_loss(mix_logits, mixed_gt_label)
                # mix_clf_loss = self.clf_loss(mix_logits, target_pred_mix) * lam + self.clf_loss(mix_logits, gen_label[index]) * (1. - lam)

                # target_pred = F.softmax(target_logits, dim=1).detach()
                # max_prob, pred_u = torch.max(target_pred, dim=-1)
                # max_prob, pred_u = torch.max(target_pred_mix.detach(), dim=-1)
                # mix_clf_loss = torch.mean(F.cross_entropy(mix_logits, pred_u, reduction='none')) * lam \
                #                 + F.cross_entropy(mix_logits, gen_label) * (1. - lam)
                # mix_logits_gini = F.softmax(mix_logits, dim=-1)
                # target_pred_clip = F.softmax(target_clip_logits, dim=-1)
                # 0.5*(target_pred+target_pred_clip.detach())
                # mix_clf_loss = self.cmkd.gini_impurity(target_pred_mix) * lam + F.cross_entropy(mix_logits, gen_label) * (1. - lam)
                # mix_clf_loss = self.clf_loss(mix_logits, mixed_gt_label)

                clf_loss = clf_loss + mix_clf_loss

        if args.mixup:
            r = np.random.rand(1)
            if args.alpha > 0 and r < args.mixup_prob:
                # generate mixed sample
                if args.alpha > 0:
                    lam = np.random.beta(args.alpha, args.alpha)
                else:
                    lam = 1
                batch_size = gen_img.size()[0]
                index = torch.randperm(batch_size)
                target_a = target_pred_mix
                target_b = gen_label[index]
                
                batch_size = target_a.size()[0]
                index = torch.randperm(batch_size)
                mix_img = lam * target_img + (1 - lam) * gen_img[index, :]
                
                # compute output
                mix_output = self.base_network.forward_features(mix_img)
                mix_logits = self.classifier_layer(mix_output)
                mix_clf_loss = self.clf_loss(mix_logits, target_a) * lam + self.clf_loss(mix_logits, target_b) * (1. - lam)
                
                clf_loss = clf_loss + mix_clf_loss
        return clf_loss, transfer_loss

    def get_parameters(self, initial_lr=1.0):
        params=[
            {'params': self.base_network.model.visual.parameters(), 'lr': initial_lr},
            {'params': self.classifier_layer.parameters(), 'lr': self.args.multiple_lr_classifier * initial_lr}
        ]
        return params

    def predict(self, x):
        features = self.base_network.forward_features(x)
        logit = self.classifier_layer(features)
        return logit

    def clip_predict(self, x):
        logit = self.base_network(x)
        return logit