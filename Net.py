import torch
import torch.nn as nn
from torch.nn.modules.activation import Sigmoid

class Net(nn.Module):
    
    def __init__(self, classes):
        """accepts parameter for # of classes"""
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