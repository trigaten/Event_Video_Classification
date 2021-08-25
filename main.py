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
the LSTM in the model.
"""

import torch
from EADataset import EADataset
from torch.utils.data import DataLoader
from Net import Net
from torch.utils.tensorboard import SummaryWriter
from train import train

# log values to tensorboard during training
writer = SummaryWriter("tensorboard_data/MODEL")

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # convergence seems to occur around 60 epochs
    EPOCHS = 100
    # initialize dataset and dataloader
    dataset = EADataset('train', device=device, do_slice=True, do_common=True)
    # batch size 1 so load 1 video at a time
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # initialize network
    net = Net(len(dataset.actions), device=device).to(device)

    train(dataloader, net, epochs=EPOCHS, writer=writer, model_save_path="MODEL")
   