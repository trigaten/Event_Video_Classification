# 1. train on the 64 dataset videos, and then test on the 15 event videos, extract accuracy, then test on the 15 model videos and extract accuracy. 
# 2. Train on 64 videos + 10 event videos (2 people all actions) and test on last person and all 15 model videos 
# 3. Train on 64 videos + 10 model videos and test on last person and all 15 event videos. 
# 4. Pick either model or event and one person. Train on all other videos (89 in total) and test on the chosen videos.

import torch
from EADataset import EADataset
from Net import Net

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

def test(model_path, dataset, conf_mat_save_path):
    model = Net(5, "cpu")
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    print("testing")
    pred = []
    true = []
    # perform inferences for every sample and log model performance
    correct = 0
    for index, (video, exp) in enumerate(dataset):
        exp = exp.item()
        print(video.shape)

        out = model(video)
        # get prediction for last frame of video
        last = out[len(out)-1]
        print(last)
        pred_index = torch.argmax(last).item()
        pred.append(pred_index)
        true.append(exp)
        print("Predicted class: " + str(pred_index)  + " Actual class: " + str(exp))
        if pred_index == exp:
            correct+=1

    print(str(correct) + " out of " + str(len(dataset)))

    conf_matrix = confusion_matrix(y_true=true, y_pred=pred)

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):    
        for j in range(conf_matrix.shape[1]):        
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large') 
            plt.xlabel('Predictions', fontsize=18)
            plt.ylabel('Actuals', fontsize=18)
            plt.title('Confusion Matrix', fontsize=18)
    # plt.show()

    fig.savefig(conf_mat_save_path)

if __name__ == "__main__":
    test("64vids/64vids_model", EADataset("data/model_videos", device="cpu"), "ttffffft")