# Event_Video_Classification

CNN+LSTM based event video classification with Pytorch.

## 1. Data
### General
The dataset consists of data collected with an event camera then synthezised into rgb images. Only the r and g channels contain data, since they represent the positive and negative changes in pixel brightness. When videos are loaded in as tensors, the r channel is removed. Contact jsengup1@jhu.edu for data. 
### Stats
The dataset consists of 64 videos, each 400 frames long, and recorded in two different locations with static backgrounds. They are approximately evenly distributed across 5 classes: ['walk_facing_forward_N_S', 'walk_facing_sideways_W_E', 'walk_in_place_N', 'walk_pivot_NE_SW', 'walk_pivot_NW_SE']

## 2. Model
The model can be found in `Net.py`

### 2.1 Architecture

3 layers of convolutions + max pooling

One LSTM layer with a 1200 node hidden layer

2 fully connected layers with sigmoid activations

### 2.2 Reasoning
CNNs are a common algorithm for deal with image classification. They are used here since we are dealing with a stream of images. The LSTM layer is necessary so the model can remember what has happened over previous frames. 

## 3. Training
### 3.1 Setup
In order to train/test the model, the data must be separated into a train/ and test/ folder. I have trained the model using a 54-10 split.

### 3.2 Train the model
Run `python main.py` to train the model. This will save the model every 10 epochs and log loss to tensorboard. Check within `main.py` to change the output paths.

### 3.3 Training Procedure
Videos are trained on one at a time, treating their frames as a batch. See `train.py` for a more detailed explanation.
Optimizer: Adam
Loss: Categorical Crossentropy (with softmax)

### 3.4 Validation
...

### 3.5 Stats

Currently very small dataset. Training using all transforms in `EADataset.py` has very large loss spikes.

98% accuracy with LOO cross validation

90% accuracy with regular train test configuration, using 10 samples in test set

40% accuracy on small set (5 samples) of videos extremely out of train distribution (shot on different camera/light conditions/frame size/quantity of action frames)
