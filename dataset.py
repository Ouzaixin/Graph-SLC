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

class OneDataset(Dataset):
    def __init__(self, file = config.train_file, name = "train"):
        self.name = name
        self.file_name = read_list(file)
        self.length_dataset = len(self.file_name)
        # self.data = pd.read_csv("data_info/info.csv",encoding = "ISO-8859-1")

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        name = self.file_name[index % self.length_dataset] + ".nii"
        # MRI_path = os.path.join(config.whole_MRI, MRI_name)
        # MRI = nifti_to_numpy(MRI_path) if os.path.exists(MRI_path) else np.zeros((1, 1))

        # FDG_path = os.path.join(config.whole_FDG, MRI_name)
        # FDG = nifti_to_numpy(FDG_path) if os.path.exists(FDG_path) else np.zeros((1, 1))

        # AV45_path = os.path.join(config.whole_AV45, MRI_name)
        # AV45 = nifti_to_numpy(AV45_path) if os.path.exists(AV45_path) else np.zeros((1, 1))

        # Tau_path = os.path.join(config.whole_Tau, MRI_name)
        # Tau = nifti_to_numpy(Tau_path) if os.path.exists(Tau_path) else np.zeros((1, 1))

        return name
