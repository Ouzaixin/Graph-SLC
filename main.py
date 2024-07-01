import os
import config
import numpy as np
import csv
import pickle
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import KernelDensity
import torch.nn.functional as F
from model import *
from dataset import *
from utils import *

os.environ['CUDA_VISIBLE_DEVICES'] ='0,1,2,3'
import warnings
warnings.filterwarnings("ignore")

# "0": NC, "1": MCI, and "2": AD
subject_modalities = {
    'subject1': {
        'modalities': ["path_of_modality_1", "path_of_modality_2", "path_of_modality_3", None],
        'label': 0
    },
    'subject2': {
        'modalities': ["path_of_modality_1", None, "path_of_modality_3", None],
        'label': 1
    },
    'subject3': {
        'modalities': ["path_of_modality_1", "path_of_modality_2", None, None],
        'label': 2
    },
    'subject4': {
        'modalities': [None, "path_of_modality_2", "path_of_modality_3", None],
        'label': 1
    }
}

def compute_nearest_neighbor_graph(subject_names):
    G = {name: None for name in subject_names}

    for i, si_name in enumerate(subject_names):
        si = subject_modalities[si_name]['modalities']
        min_distance = float('inf')
        nearest_neighbor = None
        for j, sj_name in enumerate(subject_names):
            if i != j:
                sj = subject_modalities[sj_name]['modalities']
                shared_modalities = [k for k in range(len(si)) if si[k] is not None and sj[k] is not None]
                if shared_modalities:
                    distance = sum(F.mse_loss(si[k], sj[k]) for k in shared_modalities) / len(shared_modalities)
                    if distance < min_distance:
                        min_distance = distance
                        nearest_neighbor = sj_name
        if nearest_neighbor is not None:
            G[si_name] = nearest_neighbor

    return G

def comput_KDE(subject_names):
    subjects = np.array([subject_modalities[name]['label'] for name in subject_names])
    kde_subjects = KernelDensity(kernel='gaussian').fit(subjects.reshape(-1, 1))
    log_dens_subjects = kde_subjects.score_samples(subjects.reshape(-1, 1))
    return log_dens_subjects

def compute_distribution_consistency_loss(log_dens_subjects, latent_dict):
    latent_reps = np.array([latent_dict[name].cpu().detach().numpy() for name in latent_dict])
    clustering = SpectralClustering(n_clusters=config.num_clusters, affinity='nearest_neighbors', assign_labels='kmeans').fit(latent_reps)
    cluster_labels = clustering.labels_
    kde_cluster_labels = KernelDensity(kernel='gaussian').fit(cluster_labels.reshape(-1, 1))
    log_dens_cluster_labels = kde_cluster_labels.score_samples(cluster_labels.reshape(-1, 1))
    Ld = 0
    for log_dens_si, log_dens_zi in zip(log_dens_subjects, log_dens_cluster_labels):
        Ld += log_dens_si * (log_dens_si - log_dens_zi)
    
    return Ld / len(latent_dict), cluster_labels

def train(i):
    model = Reconstruction().to(config.device)
    opt_model = optim.AdamW(model.parameters(), lr=config.learning_rate)
    MAE = nn.L1Loss()
    average_now = 0

    # Assign each subject a subject-specific latent representation
    subject_names = read_list(config.whole)
    latent_dict = {}
    for idx, t1_name in enumerate(subject_names):
        vector = torch.nn.Parameter(torch.tensor(xavier_init(1, 1728), requires_grad=True))
        latent_dict[t1_name] = vector
    opt_latent = optim.AdamW(latent_dict.values(), lr=config.learning_rate)
    G = compute_nearest_neighbor_graph(subject_names)

    train_list = train_split(task = config.whole, name = "train", fold = i)
    latent_train = {name: latent_dict[name] for name in train_list}
    log_dens_train = comput_KDE(train_list)

    validation_list = train_split(task = config.whole, name = "validation", fold = i)
    latent_validation = {name: latent_dict[name] for name in validation_list}
    test_list = train_split(task = config.whole, name = "test", fold = i)
    latent_test = {name: latent_dict[name] for name in test_list}

    for epoch in range(config.epochs):
        lossfile = open("result/"+str(config.exp)+"loss_curve.csv", 'a+',newline = '')
        writer = csv.writer(lossfile)
        if epoch == 0:
            writer.writerow(["Epoch","Reconstruction_loss"])

        dataset = OneDataset(task = config.whole, name = "train", fold = str(i))
        length = dataset.length_dataset
        loader = DataLoader(dataset,batch_size=config.batch_size,shuffle=False,num_workers=config.numworker,pin_memory=True,drop_last=True)
        loop = tqdm(loader, leave=True)

        loss_epoch = 0
        for idx, (MRI, FDG, AV45, Tau, MRI_name) in enumerate(loop):
            x = latent_dict[MRI_name[0]]
            rec_MRI, rec_FDG, rec_AV45, rec_Tau = model(x)

            loss = 0
            if not torch.all(MRI == 0):
                rec_MRI = rec_MRI.cpu().numpy()
                loss += MAE(rec_MRI, MRI)
                
            if not torch.all(FDG == 0):
                rec_FDG = rec_FDG.cpu().numpy()
                loss += MAE(rec_FDG, FDG)

            if not torch.all(AV45 == 0):
                rec_AV45 = rec_AV45.cpu().numpy()
                loss += MAE(rec_AV45, MRI)

            if not torch.all(Tau == 0):
                rec_Tau = rec_Tau.cpu().numpy()
                loss += MAE(rec_Tau, Tau)
            
            nearest_latent = latent_dict[G.get(MRI_name[0])]
            loss += MAE(x, nearest_latent)
            dist_loss, cluster_labels = compute_distribution_consistency_loss(log_dens_train, latent_train)
            loss += 0.1*dist_loss

            opt_model.zero_grad()
            opt_latent.zero_grad()
            loss.backward()
            opt_model.step()
            opt_latent.step()

        writer.writerow([epoch+1, loss_epoch])
        lossfile.close()

        for _ in range(config.epochs):
            dataset = OneDataset(task = config.whole, name = "validation", fold = str(i))
            length = dataset.length_dataset
            loader = DataLoader(dataset,batch_size=config.batch_size,shuffle=False,num_workers=config.numworker,pin_memory=True,drop_last=True)
            loop = tqdm(loader, leave=True)

            loss_epoch = 0
            for idx, (MRI, FDG, AV45, Tau, MRI_name) in enumerate(loop):
                x = latent_dict[MRI_name[0]]
                rec_MRI, rec_FDG, rec_AV45, rec_Tau = model(x)

                loss = 0
                if not torch.all(MRI == 0):
                    rec_MRI = rec_MRI.cpu().numpy()
                    loss += MAE(rec_MRI, MRI)
                    
                if not torch.all(FDG == 0):
                    rec_FDG = rec_FDG.cpu().numpy()
                    loss += MAE(rec_FDG, FDG)

                if not torch.all(AV45 == 0):
                    rec_AV45 = rec_AV45.cpu().numpy()
                    loss += MAE(rec_AV45, MRI)

                if not torch.all(Tau == 0):
                    rec_Tau = rec_Tau.cpu().numpy()
                    loss += MAE(rec_Tau, Tau)
                
                nearest_latent = latent_dict[G.get(MRI_name[0])]
                loss += MAE(x, nearest_latent)
                dist_loss, cluster_labels = compute_distribution_consistency_loss(log_dens_train, latent_validation)
                loss += 0.1*dist_loss

                opt_latent.zero_grad()
                loss.backward()
                opt_latent.step()

        labels = np.array([subject_modalities[name]['label'] for name in validation_list])
        average = indicators(labels, cluster_labels, epoch, "validation")
        
        if average > average_now:
            average_now = average
            save_checkpoint(model, opt_model, filename=config.CHECKPOINT_model)

            for _ in range(config.epochs):
                dataset = OneDataset(task = config.whole, name = "test", fold = str(i))
                length = dataset.length_dataset
                loader = DataLoader(dataset,batch_size=config.batch_size,shuffle=False,num_workers=config.numworker,pin_memory=True,drop_last=True)
                loop = tqdm(loader, leave=True)

                loss_epoch = 0
                for idx, (MRI, FDG, AV45, Tau, MRI_name) in enumerate(loop):
                    x = latent_dict[MRI_name[0]]
                    rec_MRI, rec_FDG, rec_AV45, rec_Tau = model(x)

                    loss = 0
                    if not torch.all(MRI == 0):
                        rec_MRI = rec_MRI.cpu().numpy()
                        loss += MAE(rec_MRI, MRI)
                        
                    if not torch.all(FDG == 0):
                        rec_FDG = rec_FDG.cpu().numpy()
                        loss += MAE(rec_FDG, FDG)

                    if not torch.all(AV45 == 0):
                        rec_AV45 = rec_AV45.cpu().numpy()
                        loss += MAE(rec_AV45, MRI)

                    if not torch.all(Tau == 0):
                        rec_Tau = rec_Tau.cpu().numpy()
                        loss += MAE(rec_Tau, Tau)
                    
                    nearest_latent = latent_dict[G.get(MRI_name[0])]
                    loss += MAE(x, nearest_latent)
                    dist_loss, cluster_labels = compute_distribution_consistency_loss(log_dens_train, latent_test)
                    loss += 0.1*dist_loss

                    opt_latent.zero_grad()
                    loss.backward()
                    opt_latent.step()

            labels = np.array([subject_modalities[name]['label'] for name in test_list])
            average = indicators(labels, cluster_labels, epoch, "test")

if __name__ == '__main__':
    #utils.seed_torch()
    for i in range(5):
        train(i)