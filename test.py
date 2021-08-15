import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.transforms.transforms import Grayscale
from EADataset import EADataset
from trainnow import Model

model = Model(5)

model.load_state_dict(torch.load("model"))
model.eval()

dataset = EADataset("test")

correct = 0
for index, (video, exp) in enumerate(dataset):
    video, exp = dataset[index]
    exp = exp.item()
    out = model(video)
    # get prediction for last frame of video
    last = out[len(out)-1]
    pred_index = torch.argmax(last).item()
    print("Predicted class: " + str(pred_index)  + " Actual class: " + str(exp))
    if pred_index == exp:
        correct+=1

print(str(correct) + " out of " + str(len(dataset)))
