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
    def __init__(self, root_dir, device, video_paths = None, do_slice=False, do_common=False):
        self.device = device
        # a list containing the path of every image
        if video_paths is None:
            self.video_paths = self.get_video_paths(root_dir)
        else:
            self.video_paths = video_paths

        self.root_dir = root_dir
        self.do_slice = do_slice
        self.do_common = do_common
        # implement randomness myself so have option to change assosciated action
        self.hor_flip = transforms.RandomHorizontalFlip(p=1.0)
        self.rand_persp = transforms.RandomPerspective(distortion_scale=0.4, p=0.4)
        self.actions = ['walk_facing_forward_N_S', 'walk_facing_sideways_W_E', 'walk_in_place_N', 'walk_pivot_NE_SW', 'walk_pivot_NW_SE']
    
    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        """indexing method"""
        path = self.video_paths[idx]
        vframes, _, _ = IO.read_video(path)
        # remove r channel: only g and b channels contain info
        vframes = torch.narrow(vframes, 3, 1, 2)

        vframes = vframes.to(self.device)

        # changes tensor to (frames, channels, height, width)
        vframes = vframes.permute(0, 3, 1, 2)

        # ensure correct size
        vframes = TF.resize(vframes, [420, 560])

        if self.do_slice:
            vframes = self.half(vframes)

        action = self.path_crop(path)

        if self.do_common:
            vframes, action = self.common_transform(vframes, action)

        index = torch.tensor(self.actions.index(action), device=self.device)

        return vframes.float(), index.long()


    def common_transform(self, vframes, action):
        """Applies a series of common transforms to the video frame. The action may
        change as a result of some of the transforms."""

        # This section implements a random horizontal flip on *some* of the video classes

        # action does not change due to horizontal flip
        if action == 'walk_facing_forward_N_S' or action == 'walk_in_place_N':
            if random.random() > 0.5:
                vframes = self.hor_flip(vframes)

        # action *does* change due to horizontal flip
        elif action == 'walk_pivot_NE_SW':
            if random.random() > 0.5:
                vframes, action = self.hor_flip(vframes), 'walk_pivot_NW_SE'

        elif action == 'walk_pivot_NW_SE':
            if random.random() > 0.5:
                vframes, action = self.hor_flip(vframes), 'walk_pivot_NE_SW'


        # this section randomly reverses the video if makes sense to
        if random.random() > 0.5 and action != 'walk_pivot_NE_SW' and action != 'walk_pivot_NW_SE':
            vframes = torch.flip(vframes, ([0]))

        # randomly change perspective
        if random.random() > 0.5:
            vframes = self.rand_persp(vframes)

        if random.random() > 0.5:
            h = vframes.shape[2]
            new_h = random.randint(int(h/3), h)
            w = vframes.shape[3]
            new_w = random.randint(int(w/3), w)

            vframes = TF.resize(vframes, (new_h, new_w))

            h_to_fill = h - new_h
            top_pad = random.randint(0, h_to_fill)
            w_to_fill = w - new_w
            left_pad = random.randint(0, w_to_fill)
            vframes = TF.pad(vframes, (left_pad, top_pad, w_to_fill-left_pad, h_to_fill-top_pad))
        return vframes, action
        
    def half(self, vframes):
        """with 50% probability, removes every other frame in a video. Randomly removes even or odd indexed frames"""
        if random.random() > 0.5:
            indices = torch.arange(0, len(vframes), 2, device=self.device)
            
            if random.random() > 0.5:
                indices+=1

            return torch.index_select(vframes, 0, indices)
        return vframes

    def path_crop(self, path):
        """Base on CC_#_action naming convention (character, character _ single digit _ ...) of videos, isolates the name of the action"""
        return path[6+len(self.root_dir):-24]

    def get_video_paths(self, dir):
        """reads through train directory and creates a list containing the paths of all of the videos"""
        video_paths = []
        for local_video_path in os.listdir(dir):
            video_path = os.path.join(dir, local_video_path)
            video_paths.append(video_path)
        return video_paths

if __name__ == '__main__':
    dataset = EADataset("train", 'cpu', do_slice=True, do_common=True)
    v, a = dataset[4]
    print(a)
    print(v.shape)
    v = v.permute(0, 2, 3, 1)
    r = torch.zeros([v.shape[0], v.shape[1], v.shape[2], 1])
    v = torch.cat((r, v), 3)      
    print(v.shape)  
    IO.write_video("video.avi", v.cpu().detach().numpy(), 30.0)