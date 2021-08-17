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
import torch.nn as nn
import torch.optim as optim
from EADataset import EADataset
from torch.utils.data import DataLoader
from Net import Net
from torch.utils.tensorboard import SummaryWriter

# log values to tensorboard during training
writer = SummaryWriter()

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # convergence seems to occur around 60 epochs
    EPOCHS = 100
    # initialize dataset and dataloader
    dataset = EADataset('train',  device=device, do_greyscale=True, do_slice=False)
    # batch size 1 so load 1 video at a time
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # initialize network
    net = Net(len(dataset.actions), device=device).to(device)

    # initialize loss function and optimizer
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=10e-4)

    # keep track of number of videos that have been trained on
    video_num = 0

    for epoch in range(EPOCHS):
        for index, sample in enumerate(dataloader):
            # exp is expected class
            video, exp = sample
            # remove first dim from video
            video = video.squeeze(0)
            
            # remove first 2 dims from exp
            exp = exp.squeeze(0)
            exp = exp.squeeze(0)

            # make exp into a 1d tensor which contains len(video) copies of exp
            # this is necessary for the loss function which accepts all the video frame
            # output tensors and expects a ground truth value for each frame
            exp = exp.repeat(len(video))

            # push video through neural net
            out = net(video)

            # compute loss
            loss_out = loss(out, exp)

            print(loss_out)

            # accumulate gradients
            loss_out.backward()

            # perform SGD
            optimizer.step()

            # clear grads
            optimizer.zero_grad()

            # log loss
            writer.add_scalar("Loss/train", loss_out, video_num)

            writer.flush()
            video_num+=1

        # save model every epoch
        torch.save(net.state_dict(), "dropout+no_slice_model")