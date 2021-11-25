from EADataset import EADataset
from torch.utils.data import DataLoader
from Net import Net
from train import train
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from statistics import mean 
import torch
import os


device = 'cpu'
# models generally converge at 60 epochs
EPOCHS = 20
loss_func = CrossEntropyLoss()
global_dataset = EADataset("data/event_camera_videos", device)
extra_dataset = EADataset("data/model_videos", device)


paths = global_dataset.video_paths
extra_paths = extra_dataset.video_paths
# append the last 10
for x in extra_paths[:5]:
    paths.append(x)

dataset = EADataset("", device, video_paths=paths, do_slice=False, do_common=False)

model = Net(5, device)
model.load_state_dict(torch.load("w10model_model", map_location=torch.device(device)))
model.eval()

# perform inferences for every sample and log model performance
correct = 0
for index, (video, exp) in enumerate(dataset):
    exp = exp.item()
    print(video.shape)

    out = model(video)
    # get prediction for last frame of video
    last = out[len(out)-1]
    print(last.detach().numpy())
    pred_index = torch.argmax(last).item()
    print("Predicted class: " + str(pred_index)  + " Actual class: " + str(exp))
    if pred_index == exp:
        correct+=1

print(str(correct) + " out of " + str(len(dataset)))

