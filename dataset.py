import os
from torch.utils.data import Dataset
import numpy as np
import config
import nibabel as nib
import pandas as pd
import torch

def read_list(file):
    file=open(file,"r")
    S=file.read().split()
    p=list(str(i) for i in S)
    return p

def nifti_to_numpy(file):
    data = nib.load(file).get_fdata()
    data = data.astype(np.float32)
    return data

def train_split(file,fold,name):
    file=open(file,"r")
    S=file.read().split()
    p=list(str(i) for i in S)
    x=int(len(p)/5)
    if name == "train":
        if fold == "0":
            p = p[:x*3]
        elif fold == "1":
            p = p[x:x*4]
        elif fold == "2":
            p = p[x*2:]
        elif fold == "3":
            p = p[x*3:]+p[:x]
        elif fold == "4":
            p = p[x*4:]+p[:x*2]
    elif name == "validation":
        if fold == "0":
            p = p[x*3:x*4]
        elif fold == "1":
            p = p[x*4:]
        elif fold == "2":
            p = p[:x]
        elif fold == "3":
            p = p[x:x*2]
        elif fold == "4":
            p = p[x*2:x*3]
    elif name == "test":
        if fold == "0":
            p = p[x*4:]
        elif fold == "1":
            p = p[:x]
        elif fold == "2":
            p = p[x:x*2]
        elif fold == "3":
            p = p[x*2:x*3]
        elif fold == "4":
            p = p[x*3:x*4]
    return p

class OneDataset(Dataset):
    def __init__(self, task = config.whole, name = "train", fold = "0"):
        self.name = name
        self.MRI = train_split(task,fold,name)
        self.length_dataset = len(self.MRI)
        self.data = pd.read_csv("data_info/info.csv",encoding = "ISO-8859-1")

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        MRI_name = self.MRI[index % self.length_dataset] + ".nii"
        MRI_path = os.path.join(config.whole_MRI, MRI_name)
        MRI = nifti_to_numpy(MRI_path)

        FDG_path = os.path.join(config.whole_FDG, MRI_name)
        FDG = nifti_to_numpy(FDG_path) if os.path.exists(FDG_path) else np.zeros((1, 1))

        AV45_path = os.path.join(config.whole_AV45, MRI_name)
        AV45 = nifti_to_numpy(AV45_path) if os.path.exists(AV45_path) else np.zeros((1, 1))

        Tau_path = os.path.join(config.whole_Tau, MRI_name)
        Tau = nifti_to_numpy(Tau_path) if os.path.exists(Tau_path) else np.zeros((1, 1))

        return MRI, FDG, AV45, Tau, MRI_name
