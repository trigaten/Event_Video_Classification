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
            nn.Linear(1200, 600),
            nn.Tanh(),
            nn.Linear(600, self.classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x.apply_(self.CNN)

        CNN_out = self.CNN(x)
        # print("CONV")
        # print(CNN_out.shape)
        flat = torch.flatten(CNN_out, 1)
        # print("FLAT")
        # print(flat.shape)
        # unsqueeze to add batch dimension
        flat = torch.unsqueeze(flat, 0)
        # print(flat.shape)
        lstm_out, _ = self.LSTM(flat)
        # print("LSTM")
        # print(lstm_out.shape)
        lin_out = self.Linear(lstm_out)
        # print("lin_out")
        # print(lin_out.shape)
        return torch.squeeze(lin_out, 0)


if __name__ == '__main__':
    EPOCHS = 80
    dataset = EADataset('vids', do_transform=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    net = Model(len(dataset.actions))

    loss = nn.CrossEntropyLoss()

    # net.load_state_dict(torch.load("model2"))

    # net.eval()

    optimizer = optim.Adam(net.parameters(), lr=0.001)

    video_num = 0
    for epoch in range(EPOCHS):
        for index, sample in enumerate(dataloader):
            video, exp = sample
            video = video.squeeze(0)

            exp = exp.squeeze(0)
            exp = exp.squeeze(0)

            exp = exp.repeat(400)
            out = net(video)
            # print(out.shape)
            loss_out = loss(out, exp)


            loss_out.backward()

            optimizer.step()

            optimizer.zero_grad()

            print(loss_out)

            writer.add_scalar("Loss/train", loss_out, video_num)


            writer.flush()
            video_num+=1

        torch.save(net.state_dict(), "model4")

 # video, _ = dataset[0]
    # print(video.shape)
    # # vframes = torch.unsqueeze(vframes, 0)
    # # print(vframes.shape)
    # image = video[53]
   
    # image = image.permute(1, 2, 0)
    # print(image.shape)
    # plt.imshow(image.byte())
    # print("HI")
    # plt.show()