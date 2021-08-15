""" Dataset object for the action videos """

__author__ = "Sander Schulhoff"
__email__ = "sanderschulhoff@gmail.com"

import torch
import random

import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision.io as IO
import os
from torch.utils.data import Dataset

class EADataset(Dataset):
    def __init__(self, root_dir, do_transform=True):
        # a list containing the path of every image
        self.video_paths = self.get_video_paths(root_dir)
        self.root_dir = root_dir
        self.do_transform = do_transform
        self.grayscale = transforms.Grayscale(num_output_channels=1)
        self.actions = ['walk_facing_forward_N_S', 'walk_facing_sideways_W_E', 'walk_in_place_N', 'walk_pivot_NE_SW', 'walk_pivot_NW_SE']
    
    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        """indexing method"""
        path = self.video_paths[idx]
        vframes, _, _ = IO.read_video(path)
        vframes = vframes.permute(0, 3, 1, 2)

        if self.do_transform:
            vframes = self.transform(vframes)

        index = self.actions.index(self.path_crop(path))
        return vframes.float(), torch.LongTensor([index])

    def transform(self, vframes):
        return self.grayscale(vframes)

    def path_crop(self, path):
        return path[10:-24]

    def get_video_paths(self, dir):
        video_paths = []
        for local_video_path in os.listdir(dir):
            video_path = os.path.join(dir, local_video_path)
            video_paths.append(video_path)
        return video_paths

