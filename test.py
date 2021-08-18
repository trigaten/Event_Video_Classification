import torch
from EADataset import EADataset
from Net import Net

# load model
model = Net(5, "cpu")
model.load_state_dict(torch.load("nodropout"))
model.eval()

# load testing dataset
dataset = EADataset("test", device="cpu")

# perform inferences for every sample and log model performance
correct = 0
for index, (video, exp) in enumerate(dataset):
    exp = exp.item()
    out = model(video)
    # get prediction for last frame of video
    last = out[len(out)-1]
    print(last.detach().numpy())
    pred_index = torch.argmax(last).item()
    print("Predicted class: " + str(pred_index)  + " Actual class: " + str(exp))
    if pred_index == exp:
        correct+=1

print(str(correct) + " out of " + str(len(dataset)))
