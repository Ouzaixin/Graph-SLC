import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
learning_rate = 1e-4
batch_size = 1
numworker = 0
epochs = 1000
latent_dim = 1728
num_clusters = 3
gpus = [0]

whole_MRI = "./data/whole_MRI"
whole_FDG = "./data/whole_FDG"
whole_AV45 = "./data/whole_AV45"
whole_Tau = "./data/whole_Tau"

exp = "exp_1/"
train_file = "./data_info/train.txt"
validation_file = "./data_info/validation.txt"
test_file = "./data_info/test.txt"

CHECKPOINT_model = "result/"+exp+"model.pth.tar"
