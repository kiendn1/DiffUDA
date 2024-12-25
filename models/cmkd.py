import torch.nn as nn
import torch
import torch.nn.functional as F
from utils.tools import LambdaSheduler

def feat_prototype_distance_module(feat, prototypes, class_numbers):
    N, D = feat.shape
    feat_proto_distance = -torch.ones((N, class_numbers))
    for i in range(class_numbers):
        feat_proto_distance[:, i] = torch.norm(prototypes[i].unsqueeze(0) - feat, 2, dim=1,)
    return feat_proto_distance

class CMKD(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.lamb = LambdaSheduler(max_iter=args.max_iter)
        self.args = args

    def calibrated_coefficient(self, pred, pred_pretrained):
        distance = F.kl_div(pred.log(), pred_pretrained, reduction='none').sum(-1)
        coe = torch.exp(-distance).detach()
        return coe

    def calibrated_coefficient1(self, pred):
        epsilon = 1e-5
        H = -pred * torch.log(pred + epsilon)
        H = H.sum(dim=1)
        coe = torch.exp(-H).detach()
        return coe

    def gini_impurity(self,pred,coe=1.0):
        sum_dim = torch.sum(pred, dim=0).unsqueeze(dim=0).detach()
        return torch.sum(coe * (1 - torch.sum(pred ** 2 / sum_dim, dim=-1)))

    def regularization_term(self, target_pred_clip, source_logit_clip, source_label,lamb, **kwargs):
        if 'gen_logit_clip' in kwargs:
            return self.args.lambda2*F.cross_entropy(source_logit_clip, source_label) + \
                self.args.lambda2*F.cross_entropy(kwargs['gen_logit_clip'], kwargs['gen_label']) + \
                self.args.lambda3*lamb*self.gini_impurity(target_pred_clip)
        else:
            return self.args.lambda2*F.cross_entropy(source_logit_clip, source_label) + \
                self.args.lambda3*lamb*self.gini_impurity(target_pred_clip)

    def forward(self, target_logit, target_logit_clip, source_logit_clip, source_label, label_set=None, prototypes=None, target_feat=None, 
                use_contrastive_loss=None, gen_feat=None, gen_label=None, source_feat=None, **kwargs):
        target_pred = F.softmax(target_logit, dim=1)
        target_pred_clip = F.softmax(target_logit_clip,dim=-1)
        coe = self.calibrated_coefficient(target_pred, target_pred_clip)
        target_pred_mix = 0.5*(target_pred+target_pred_clip.detach())
        # with torch.no_grad():
        #     if prototypes is not None:
        #         proto_temperature = 8
        #         # sorted_keys = sorted(prototypes.keys())
        #         # prototypes_tensor = torch.stack([prototypes[key] for key in sorted_keys], dim=0)
        #         # normalize_feat_data = torch.nn.functional.normalize(target_feat, p=2, dim=1)
        #         # normalize_prototypes = torch.nn.functional.normalize(prototypes_tensor, p=2, dim=1)
        #         feat_proto_distance = feat_prototype_distance_module(target_feat, prototypes, class_numbers=65)
        #         # feat_proto_distance = torch.matmul(normalize_feat_data, normalize_prototypes.t())
        #         prob_prototype = F.softmax(-feat_proto_distance * proto_temperature, dim=1)
        #         # avg_predict = 1/2*(target_pred_clip+ prob_prototype)
        #         # avg_predict = F.softmax(prob_prototype * target_pred_clip, dim=1)
        # coe = self.calibrated_coefficient(target_pred, prob_prototype)
        # target_pred_mix = 1/2*(target_pred + prob_prototype.detach())
        lamb = self.lamb.lamb()
        if label_set is not None:
            task_loss = self.args.lambda1 * lamb * self.gini_impurity(target_pred[:,label_set], coe)
            distill_loss = self.args.lambda1 * lamb * self.gini_impurity(target_pred_mix[:,label_set], 1 - coe)
            if 'gen_logit_clip' in kwargs:
                print('hi')
                reg_loss = self.regularization_term(target_pred_clip[:,label_set], source_logit_clip, source_label, lamb, 
                                                    gen_logit_clip=kwargs['gen_logit_clip'], gen_label=kwargs['gen_label'])
            else:
                reg_loss = self.regularization_term(target_pred_clip[:,label_set], source_logit_clip, source_label, lamb)
        else:
            task_loss = self.args.lambda1 * lamb * self.gini_impurity(target_pred,coe)
            distill_loss = self.args.lambda1 * lamb *self.gini_impurity(target_pred_mix,1-coe)
            if 'gen_logit_clip' in kwargs:
                reg_loss = self.regularization_term(target_pred_clip, source_logit_clip, source_label,lamb,
                                                    gen_logit_clip=kwargs['gen_logit_clip'], gen_label=kwargs['gen_label'])
            else:
                reg_loss = self.regularization_term(target_pred_clip, source_logit_clip, source_label,lamb)
        self.lamb.step()
        return task_loss + distill_loss + reg_loss, target_pred_mix
        
        # if use_contrastive_loss:
        #     TAU = 10
        #     tfeat = F.normalize(target_feat, p=2, dim=1)
        #     Proto = F.normalize(prototypes.detach(), p=2, dim=1)
        #     tlogits = tfeat.mm(Proto.permute(1, 0).contiguous())
        #     tlogits = tlogits / TAU
            
        #     gfeat = F.normalize(gen_feat, p=2, dim=1)
        #     glogits = gfeat.mm(Proto.permute(1, 0).contiguous())
        #     glogits = glogits / TAU
            
        #     sfeat = F.normalize(source_feat, p=2, dim=1)
        #     slogits = sfeat.mm(Proto.permute(1, 0).contiguous())
        #     slogits = slogits / TAU
            
        #     ce_criterion = nn.CrossEntropyLoss()
        #     contrastive_loss = ce_criterion(tlogits, target_pred_mix.detach()) + ce_criterion(glogits, gen_label) + ce_criterion(slogits, source_label)
        #     self.lamb.step()
        #     return task_loss + distill_loss + contrastive_loss, target_pred_mix
        # else:
        #     self.lamb.step()
        #     return task_loss + distill_loss + reg_loss, target_pred_mix

