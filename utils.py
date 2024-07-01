import os
import csv
import torch
import config
import random
import numpy as np
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score,recall_score, precision_score, confusion_matrix

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def specificity(y_true, y_pred, labels=None, pos_label=1, average='macro', sample_weight=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels, sample_weight=sample_weight)
    true_negatives = cm.sum(axis=1) - cm.diagonal()
    false_positives = cm.sum(axis=0) - cm.diagonal()
    false_negatives = np.sum(cm, axis=1) - cm.diagonal()
    true_positives = np.sum(cm) - false_positives - false_negatives - cm.diagonal()

    specificity_per_class = true_negatives / (true_negatives + false_positives + 1e-6)

    if average == 'macro':
        return np.mean(specificity_per_class)
    elif average == 'micro':
        total_true_negatives = np.sum(true_negatives)
        total_false_positives = np.sum(false_positives)
        overall_specificity = total_true_negatives / (total_true_negatives + total_false_positives + 1e-6)
        return overall_specificity
    elif average == 'weighted':
        class_counts = np.sum(cm, axis=1)
        weights = class_counts / np.sum(class_counts)
        weighted_specificity = np.sum(specificity_per_class * weights)
        return weighted_specificity
    else:
        raise ValueError("Invalid value for 'average'. Use 'macro', 'micro', or 'weighted'.")

def indicators(label,predict,epoch,name="nonspecific"):
    csvfile = open("result/"+str(config.exp)+str(name)+".csv", 'a+',newline = '')
    writer = csv.writer(csvfile)
    if epoch == 0:
        writer.writerow(["Epoch","accuracy","precision","recall","F1_score","specificity_score","Roc_auc_score"])

    accuracy = accuracy_score(label, predict)
    specificity_score  = specificity(label, predict, average='macro')
    precision = precision_score(label, predict, average='weighted')
    recall = recall_score(label, predict, average='weighted')
    F1_score = f1_score(label, predict, average='weighted')
    average = (accuracy + specificity_score + precision + recall + F1_score)/5
    writer.writerow([epoch+1,accuracy,precision,recall,F1_score,specificity_score])
    csvfile.close()
    return average