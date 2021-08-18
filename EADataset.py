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
    def __init__(self, root_dir, device, video_paths = None, do_greyscale=True, do_slice=False):
        self.device = device
        # a list containing the path of every image
        if video_paths is None:
            self.video_paths = self.get_video_paths(root_dir)
        else:
            self.video_paths = video_paths

        self.root_dir = root_dir
        self.do_greyscale = do_greyscale
        self.do_slice = do_slice
        self.grayscale = transforms.Grayscale(num_output_channels=1)
        self.actions = ['walk_facing_forward_N_S', 'walk_facing_sideways_W_E', 'walk_in_place_N', 'walk_pivot_NE_SW', 'walk_pivot_NW_SE']
    
    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        """indexing method"""
        path = self.video_paths[idx]
        vframes, _, _ = IO.read_video(path)
        vframes = vframes.to(self.device)
        vframes = vframes.permute(0, 3, 1, 2)

        if self.do_greyscale:
            vframes = self.grayscale(vframes)

        if self.do_slice:
            vframes = self.half(vframes)

        index = torch.tensor(self.actions.index(self.path_crop(path)), device=self.device)

        return vframes.float(), index.long()

    def half(self, vframes):
        if random.random() > 0.5:
            indices = torch.arange(0, len(vframes), 2, device=self.device)
            
            if random.random() > 0.5:
                indices+=1

            return torch.index_select(vframes, 0, indices)
        return vframes

    def path_crop(self, path):
        return path[6+len(self.root_dir):-24]

    def get_video_paths(self, dir):
        video_paths = []
        for local_video_path in os.listdir(dir):
            video_path = os.path.join(dir, local_video_path)
            video_paths.append(video_path)
        return video_paths

if __name__ == '__main__':
    dataset = EADataset("train", 'cuda')
    print(len(dataset))