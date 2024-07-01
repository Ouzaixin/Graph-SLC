import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
learning_rate = 1e-4
batch_size = 1
numworker = 0
epochs = 40
gpus = [0]

whole_MRI = "./data/whole_MRI"
whole_FDG = "./data/whole_FDG"
whole_AV45 = "./data/whole_AV45"
whole_Tau = "./data/whole_Tau"

exp = "exp_1/"
whole = "./data_info/whole.txt"
num_clusters = 3

CHECKPOINT_model = "result/"+exp+"model.pth.tar"