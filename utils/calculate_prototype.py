import numpy as np
import torch.nn.functional as F
import torch
from utils import data_loader
def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
       m.eval()

def init_prototype(base_network, args, dir_data):
    ## create data loader
    dataloader, _ = data_loader.load_data(
        args, dir_data, 32, infinite_data_loader=False, train=True, num_workers=args.num_workers)

    # begin training
    base_network.apply(fix_bn)
    features_list = []
    labels_list = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.cuda(), labels.cuda()
            features = base_network.forward_features(inputs)
        
            features_list.append(features.cpu())
            labels_list.append(labels.cpu())

            features_all = torch.cat(features_list, dim=0)
            labels_all = torch.cat(labels_list, dim=0)

    unique_labels = torch.unique(labels_all)

    prototypes = {}

    for label in unique_labels:
        class_features = features_all[labels_all == label]
        class_prototype = class_features.mean(dim=0)
        prototypes[label.item()] = class_prototype.cuda()
    
    sorted_keys = sorted(prototypes.keys())
    prototypes = torch.stack([prototypes[key] for key in sorted_keys], dim=0)
    
    return prototypes

def calculate_mean_vector_by_output(prototypes, feat_cls, outputs, alpha):
    max_prob, outputs_argmax = torch.max(outputs, dim=1)
    outputs_pred = outputs_argmax
    unique_labels = torch.unique(outputs_pred)
    with torch.no_grad():
        for label in unique_labels:
            # Extract the features corresponding to the current class
            class_features = feat_cls[outputs_pred == label]
            # class_features = class_features[max_prob[outputs_pred == label]>=0.7]
            # if class_features.shape[0] != 0:
            mean_features = class_features.mean(dim=0)
            if label.item() in prototypes:
                prototypes[label.item()] = alpha * prototypes[label.item()] + (1 - alpha) * mean_features
    return prototypes
    
def calculate_mean_vector_by_label(prototypes, feat_cls, labels, alpha):
    with torch.no_grad():
        unique_labels = torch.unique(labels)
        for label in unique_labels:
            # Extract the features corresponding to the current class
            class_features = feat_cls[labels == label]
            mean_features = class_features.mean(dim=0)
            if label.item() in prototypes:
                prototypes[label.item()] = alpha * prototypes[label.item()] + (1 - alpha) * mean_features
    return prototypes