"""prints a list of the classes present in the train data"""
import os
d = {}
for local_video_path in os.listdir("train"):
    d[local_video_path[5:-24]] = ""

print(list(d))