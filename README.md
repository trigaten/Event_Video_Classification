# Event_Video_Classification

CNN+LSTM based event video classification. The event videos contain frames from an event based camera.

### runs contains tensorboard loss data collected during the training procedure

### stats

Currently very small dataset

98% accuracy with LOO cross validation

90% accuracy with regular train test configuration, using 10 samples in test set

40% accuracy on small set (5 samples) of videos extremely out of train distribution (shot on different camera/light conditions/frame size/quantity of action frames)
