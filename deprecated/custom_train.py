from EADataset import EADataset
from torch.utils.data import DataLoader
from Net import Net
from train import train
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from statistics import mean 
import torch
import os

# log values to tensorboard during training
writer = SummaryWriter("w10model_vids")
# models generally converge at 60 epochs
EPOCHS = 30
loss_func = CrossEntropyLoss()
global_dataset = EADataset("data/train", 'cuda')
extra_dataset = EADataset("data/model_videos", 'cuda')
device = 'cuda'

paths = global_dataset.video_paths
extra_paths = extra_dataset.video_paths
print(len(paths))
# append the last 10
for x in extra_paths[5:]:
    print(x)
    paths.append(x)
# print(len(paths))
# exit()
print(paths)
# exit()
dataset = EADataset("", device, video_paths=paths, do_slice=True, do_common=True)

dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

net = Net(len(dataset.actions), device=device).to(device)

train(dataloader, net, EPOCHS, writer=writer, loss_log_path="Loss: ", model_save_path="w10model_model")

