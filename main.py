import os
import config
import numpy as np
import csv
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
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

def differentiable_kmeans(features, num_clusters, num_iters=10):
    centers = features[torch.randperm(features.size(0))[:num_clusters]]
    for _ in range(num_iters):
        distances = torch.cdist(features, centers)
        soft_assignments = F.softmax(-distances, dim=1)
        centers = torch.matmul(soft_assignments.t(), features) / soft_assignments.sum(dim=0, keepdim=True).t()
    return soft_assignments, centers

def differentiable_kde(soft_assignments, bandwidth=1.0, num_points=1000):
    soft_assignments = soft_assignments.view(-1, 1).to(config.device)
    x = torch.linspace(0, 2, num_points).view(-1, 1).to(config.device)
    diffs = x - soft_assignments.T
    kernel_vals = torch.exp(-0.5 * (diffs / bandwidth) ** 2) / (bandwidth * torch.sqrt(torch.tensor(2.0 * torch.pi)))
    kde_vals = kernel_vals.sum(dim=1) / len(soft_assignments)
    return x.squeeze(), kde_vals

def comput_KDE(subject_names):
    label_info = pd.read_csv("data_info/info.csv", encoding="ISO-8859-1")
    subjects = np.array([label_info[label_info['ID'] == name]['label'].values[0] for name in subject_names])
    subjects_tensor = torch.tensor(subjects, dtype=torch.float32).reshape(-1, 1)
    x, kde_vals = differentiable_kde(subjects_tensor)
    return x, kde_vals

def compute_distribution_consistency_loss(log_dens_subjects, latent_tensor):
    similarity_matrix = F.cosine_similarity(latent_tensor.unsqueeze(1), latent_tensor.unsqueeze(0), dim=2)
    D = torch.diag(similarity_matrix.sum(dim=1))
    L = D - similarity_matrix
    diag_sum = similarity_matrix.sum(dim=1) + 1e-6
    D_inv_sqrt = torch.diag(torch.pow(diag_sum, -0.5))
    small_constant = torch.full_like(D_inv_sqrt, 1e-6)
    D_inv_sqrt = torch.where(torch.isfinite(D_inv_sqrt), D_inv_sqrt, small_constant)
    L_normalized = torch.mm(torch.mm(D_inv_sqrt, L), D_inv_sqrt)

    linear_layer = torch.nn.Linear(L_normalized.shape[1], config.num_clusters).to(config.device)
    features = linear_layer(L_normalized)
    cluster_labels, centers = differentiable_kmeans(features, config.num_clusters)
    _, log_dens_cluster_labels = differentiable_kde(cluster_labels)
    log_dens_subjects_tensor = torch.tensor(log_dens_subjects, dtype=torch.float32, device=config.device, requires_grad=True)
    Ld = torch.sum(log_dens_subjects_tensor * torch.abs(log_dens_subjects_tensor - log_dens_cluster_labels))
    return Ld, cluster_labels

def train():
    model = Reconstruction().to(config.device)
    opt_model = optim.AdamW(model.parameters(), lr=config.learning_rate)

    # Assign each subject a subject-specific latent representation
    subject_names = read_list(config.whole)
    latent_dict = {}
    for t1_name in subject_names:
        latent_dict[t1_name] = Variable(xavier_init(1, 1728), requires_grad=True)
    opt_latent = optim.AdamW(latent_dict.values(), lr=config.learning_rate)

    train_list = read_list(config.train_file)
    latent_train = {name: latent_dict[name] for name in train_list}
    validation_list = read_list(config.validation_file)
    latent_validation = {name: latent_dict[name] for name in validation_list}
    test_list = read_list(config.test_file)
    latent_test = {name: latent_dict[name] for name in test_list}

    G = compute_nearest_neighbor_graph(subject_names)
    _, log_dens_train = comput_KDE(read_list(config.train_file))

    MAE = nn.L1Loss()
    average_now = 0
    for epoch in range(config.epochs):
        lossfile = open("result/"+str(config.exp)+"loss_curve.csv", 'a+',newline = '')
        writer = csv.writer(lossfile)
        if epoch == 0:
            writer.writerow(["Epoch","Reconstruction_loss"])

        dataset = OneDataset(file = config.train_file, name = "train")
        length = dataset.length_dataset
        loader = DataLoader(dataset,batch_size=config.batch_size,shuffle=True,num_workers=config.numworker,pin_memory=True,drop_last=True)
        loop = tqdm(loader, leave=True)

        loss_epoch = 0
        for idx, (name) in enumerate(loop):
            x = latent_train[name[0][:-4]]
            rec_MRI, rec_FDG, rec_AV45, rec_Tau = model(x)
            loss = 0

            MRI_path = os.path.join(config.whole_MRI, name[0])
            if os.path.exists(MRI_path):
                MRI = nifti_to_numpy(MRI_path)
                MRI = np.expand_dims(MRI, axis=0)
                MRI = np.expand_dims(MRI, axis=1)
                MRI = torch.tensor(MRI).to(config.device)
                loss += MAE(rec_MRI, MRI)
            
            FDG_path = os.path.join(config.whole_FDG, name[0])
            if os.path.exists(FDG_path):
                FDG = nifti_to_numpy(FDG_path)
                FDG = np.expand_dims(FDG, axis=0)
                FDG = np.expand_dims(FDG, axis=1)
                FDG = torch.tensor(FDG).to(config.device)
                loss += MAE(rec_FDG, FDG)

            AV45_path = os.path.join(config.whole_AV45, name[0])
            if os.path.exists(AV45_path):
                AV45 = nifti_to_numpy(AV45_path)
                AV45 = np.expand_dims(AV45, axis=0)
                AV45 = np.expand_dims(AV45, axis=1)
                AV45 = torch.tensor(AV45).to(config.device)
                loss += MAE(rec_AV45, AV45)

            Tau_path = os.path.join(config.whole_Tau, name[0])
            if os.path.exists(Tau_path):
                Tau = nifti_to_numpy(Tau_path)
                Tau = np.expand_dims(Tau, axis=0)
                Tau = np.expand_dims(Tau, axis=1)
                Tau = torch.tensor(Tau).to(config.device)
                loss += MAE(rec_Tau, Tau)
        
            nearest_latent = latent_dict[G.get(name[0])]
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
            dataset = OneDataset(file = config.validation_file, name = "validation")
            length = dataset.length_dataset
            loader = DataLoader(dataset,batch_size=config.batch_size,shuffle=True,num_workers=config.numworker,pin_memory=True,drop_last=True)
            loop = tqdm(loader, leave=True)

            loss_epoch = 0
            labels = []
            for idx, (MRI, FDG, AV45, Tau, name) in enumerate(loop):
                x = latent_validation[name[0][:-4]]
                rec_MRI, rec_FDG, rec_AV45, rec_Tau = model(x)
                loss = 0

                MRI_path = os.path.join(config.whole_MRI, name[0])
                if os.path.exists(MRI_path):
                    MRI = nifti_to_numpy(MRI_path)
                    MRI = np.expand_dims(MRI, axis=0)
                    MRI = np.expand_dims(MRI, axis=1)
                    MRI = torch.tensor(MRI).to(config.device)
                    loss += MAE(rec_MRI, MRI)
                
                FDG_path = os.path.join(config.whole_FDG, name[0])
                if os.path.exists(FDG_path):
                    FDG = nifti_to_numpy(FDG_path)
                    FDG = np.expand_dims(FDG, axis=0)
                    FDG = np.expand_dims(FDG, axis=1)
                    FDG = torch.tensor(FDG).to(config.device)
                    loss += MAE(rec_FDG, FDG)

                AV45_path = os.path.join(config.whole_AV45, name[0])
                if os.path.exists(AV45_path):
                    AV45 = nifti_to_numpy(AV45_path)
                    AV45 = np.expand_dims(AV45, axis=0)
                    AV45 = np.expand_dims(AV45, axis=1)
                    AV45 = torch.tensor(AV45).to(config.device)
                    loss += MAE(rec_AV45, AV45)

                Tau_path = os.path.join(config.whole_Tau, name[0])
                if os.path.exists(Tau_path):
                    Tau = nifti_to_numpy(Tau_path)
                    Tau = np.expand_dims(Tau, axis=0)
                    Tau = np.expand_dims(Tau, axis=1)
                    Tau = torch.tensor(Tau).to(config.device)
                    loss += MAE(rec_Tau, Tau)

                nearest_latent = latent_dict[G.get(name[0])]
                loss += MAE(x, nearest_latent)
                dist_loss, cluster_labels = compute_distribution_consistency_loss(log_dens_train, latent_validation)
                loss += 0.1*dist_loss

                opt_latent.zero_grad()
                loss.backward()
                opt_latent.step()

                labels.append(subject_modalities[name]['label'])
        cluster_labels = torch.argmax(cluster_labels, dim=1)
        average = indicators(labels, cluster_labels.detach().cpu().numpy(), epoch, "validation")

        if average > average_now:
            average_now = average
            save_checkpoint(model, opt_model, filename=config.CHECKPOINT_model)

            dataset = OneDataset(file = config.test_file, name = "test")
            length = dataset.length_dataset
            loader = DataLoader(dataset,batch_size=config.batch_size,shuffle=True,num_workers=config.numworker,pin_memory=True,drop_last=True)
            loop = tqdm(loader, leave=True)

            loss_epoch = 0
            for idx, (name) in enumerate(loop):
                x = latent_test[name[0][:-4]]
                rec_MRI, rec_FDG, rec_AV45, rec_Tau = model(x)
                loss = 0
                labels = []

                MRI_path = os.path.join(config.whole_MRI, name[0])
                if os.path.exists(MRI_path):
                    MRI = nifti_to_numpy(MRI_path)
                    MRI = np.expand_dims(MRI, axis=0)
                    MRI = np.expand_dims(MRI, axis=1)
                    MRI = torch.tensor(MRI).to(config.device)
                    loss += MAE(rec_MRI, MRI)
                
                FDG_path = os.path.join(config.whole_FDG, name[0])
                if os.path.exists(FDG_path):
                    FDG = nifti_to_numpy(FDG_path)
                    FDG = np.expand_dims(FDG, axis=0)
                    FDG = np.expand_dims(FDG, axis=1)
                    FDG = torch.tensor(FDG).to(config.device)
                    loss += MAE(rec_FDG, FDG)

                AV45_path = os.path.join(config.whole_AV45, name[0])
                if os.path.exists(AV45_path):
                    AV45 = nifti_to_numpy(AV45_path)
                    AV45 = np.expand_dims(AV45, axis=0)
                    AV45 = np.expand_dims(AV45, axis=1)
                    AV45 = torch.tensor(AV45).to(config.device)
                    loss += MAE(rec_AV45, AV45)

                Tau_path = os.path.join(config.whole_Tau, name[0])
                if os.path.exists(Tau_path):
                    Tau = nifti_to_numpy(Tau_path)
                    Tau = np.expand_dims(Tau, axis=0)
                    Tau = np.expand_dims(Tau, axis=1)
                    Tau = torch.tensor(Tau).to(config.device)
                    loss += MAE(rec_Tau, Tau)

                nearest_latent = latent_dict[G.get(name[0])]
                loss += MAE(x, nearest_latent)
                dist_loss, cluster_labels = compute_distribution_consistency_loss(log_dens_train, latent_test)
                loss += 0.1*dist_loss

                opt_latent.zero_grad()
                loss.backward()
                opt_latent.step()

                labels.append(subject_modalities[name]['label'])
            cluster_labels = torch.argmax(cluster_labels, dim=1)
            average = indicators(labels, cluster_labels.detach().cpu().numpy(), epoch, "test")

if __name__ == '__main__':
    #utils.seed_torch()
    train()
