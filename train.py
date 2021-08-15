import torch
import torch.nn as nn
from torch.nn.modules.activation import Tanh
from torch.nn.modules.activation import Sigmoid
import torch.optim as optim
from torch.utils import data
import torchvision  # .io.read_video as read_video
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.transforms.transforms import Grayscale
from EADataset import EADataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

class Model(nn.Module):
    def __init__(self, classes):
        super().__init__()

        self.classes = classes

        self.CNN = nn.Sequential(
            nn.Conv2d(1, 10, 4, stride=2),
            nn.MaxPool2d(4),
            nn.Conv2d(10, 20, 3),
            nn.MaxPool2d(3),
            nn.Conv2d(20, 30, 2),
            nn.MaxPool2d(2),
        )

        self.LSTM = nn.LSTM(2100, 1200, batch_first=True)

        self.Linear = nn.Sequential(
            nn.Linear(1200, 600, bias=False),
            nn.Sigmoid(),
            nn.Linear(600, self.classes, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # CNN output
        CNN_out = self.CNN(x)
        # flatten it
        flat = torch.flatten(CNN_out, 1)
        # unsqueeze to add batch dimension
        flat = torch.unsqueeze(flat, 0)
        # pass 0s as hidden and cell state since 
        # this is the start of a sequence of images
        h_n = torch.zeros(1, 1, 1200)
        c_n = torch.zeros(1, 1, 1200)
        # lstm output
        lstm_out, _ = self.LSTM(flat, (h_n, c_n))
        # linear layer output
        lin_out = self.Linear(lstm_out)

        return torch.squeeze(lin_out, 0)

if __name__ == '__main__':
    EPOCHS = 100
    # initialize dataset and dataloader
    dataset = EADataset('vids', do_transform=True)
    # batch size 1 so load 1 video at a time
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # initialize network
    net = Model(len(dataset.actions))

    loss = nn.CrossEntropyLoss()

    optimizer = optim.Adam(net.parameters(), lr=0.0001)


    video_num = 0
    for epoch in range(EPOCHS):
        for index, sample in enumerate(dataloader):
            video, exp = sample
            video = video.squeeze(0)

            exp = exp.squeeze(0)
            exp = exp.squeeze(0)
            exp = exp.repeat(400)
            out = net(video)

            loss_out = loss(out, exp)

            loss_out.backward()

            optimizer.step()

            optimizer.zero_grad()

            print(loss_out) 

            writer.add_scalar("Loss/train", loss_out, video_num)

            writer.flush()
            video_num+=1

        torch.save(net.state_dict(), "model8")