import sys
import os
import re

file_path = sys.argv[1]
in_place = re.compile("in_place")
walk_N_S = re.compile("walk_N_S")
walk_NE_SW = re.compile("walk_NE_SW")
walk_NW_SE = re.compile("walk_NW_SE")
walk_W_E = re.compile("walk_W_E")


def unchanged(str, str2):
    return str == str2
    
for video_path in os.listdir(file_path):
    
    new_name = re.sub(in_place, "in_place_N", video_path)
    if unchanged(video_path, new_name):
        new_name = re.sub(walk_N_S, "walk_facing_forward_N_S", video_path)
    if unchanged(video_path, new_name):
        new_name = re.sub(walk_NE_SW, "walk_pivot_NE_SW", video_path)
    if unchanged(video_path, new_name):
        new_name = re.sub(walk_NW_SE, "walk_pivot_NW_SE", video_path)
    if unchanged(video_path, new_name):
        new_name = re.sub(walk_W_E, "walk_facing_sideways_W_E", video_path)
    
    os.rename(os.path.join(file_path, video_path), os.path.join(file_path, new_name))


        