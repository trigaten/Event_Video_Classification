import torch
import torch.nn as nn
import torch.optim as optim
import torchvision#.io.read_video as read_video

class Model(nn.Module):
    def __init__(self, obs_size, act_size):
        super(Model, self.__init__())
        self.CNN = nn.Sequential(
            nn.Conv2d(3, 20, 3),
            nn.MaxPool2d(3),
            nn.Conv2d(20, 60, 2),
            nn.MaxPool2d(2),
        )

        self.LSTM = nn.LSTM()

    def forward(self, x):
        conv = self.CNN(x)
        flat = torch.flatten(conv, 1)
        out = self.LSTM(flat)
        return out

if __name__ == '__main__':
    vframes , _, _ = torchvision.io.read_video("Users\psi\Videos\Captures\pyt.p")
    print(vframes)