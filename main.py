"""
This code trains the model

Training procedure:
for each epoch:
    for each sample video in the train dataset (sampled randomly)
        push the entire video through the neural network and get a class prediction tensor for each frame
        apply loss function using the label of the video as the target class for every frame
        get average loss from loss function and perform SGD

This method basically treats each video as a batch of images since the loss is their average loss
Note, however, that the prediction of each frame/image is affected by the previous images due to
the GRU in the model.

Reads configurations from a yaml file
"""

import torch
from EADataset import EADataset
from torch.utils.data import DataLoader
from Net import Net
from torch.utils.tensorboard import SummaryWriter
from train import train
from test import test
from configuration_reader import read_datasets
import os

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for (config_name, train_dataset, train_epochs, test_dataset) in read_datasets("config.yaml"):
        if train_dataset:
            # log values to tensorboard during training
            writer = SummaryWriter(os.path.join(config_name, "tensorboard_data"))

            # initialize dataset and dataloader
            dataset = EADataset('data/train', device=device, do_slice=True, do_common=True)
            # batch size 1 so load 1 video at a time
            dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

            # initialize network
            net = Net(len(dataset.actions), device=device).to(device)
            
            train(dataloader, net, epochs=train_epochs, writer=writer, model_save_path=os.path.join(config_name, config_name + "_model"))
        
        if test_dataset:
            test(os.path.join(config_name, config_name + "_model"), test_dataset, os.path.join(config_name, config_name + "_confusion_matrix"))


        
   