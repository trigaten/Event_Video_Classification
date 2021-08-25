# Event_Video_Classification

CNN+LSTM based event video classification with Pytorch.

## 1. Data
The dataset consists of data collected with an event camera then synthezised into rgb images. Only the r and g channels contain data, since they represent the positive and negative changes in pixel brightness. When videos are loaded in as tensors, the r channel is removed. Contact jsengup1@jhu.edu for data. 

## 2. Training
### 2.1 Setup
In order to train/test the model, the data must be separated into a train/ and test/ folder. I have trained the model using a 54-10 split.
### 2.2 Train the model
Run `python main.py` to train the model. This will save the model every 10 epochs and log loss to tensorboard. Check within `main.py` to change the output paths.

### runs contains tensorboard loss data collected during the training procedure

### stats

Currently very small dataset

98% accuracy with LOO cross validation

90% accuracy with regular train test configuration, using 10 samples in test set

40% accuracy on small set (5 samples) of videos extremely out of train distribution (shot on different camera/light conditions/frame size/quantity of action frames)
