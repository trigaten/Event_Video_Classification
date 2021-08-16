import torch
from EADataset import EADataset
from Net import Net

# load model
model = Net(5)
model.load_state_dict(torch.load("model"))
model.eval()

# load testing dataset
dataset = EADataset("test")

# perform inferences for every sample and log model performance
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
