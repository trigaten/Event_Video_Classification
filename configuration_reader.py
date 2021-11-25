from torch.utils.data.dataset import Dataset
import yaml
from EADataset import EADataset
import re
from train import train
from functools import reduce 

import os
extract_instruction_reg = r"(.*) (\d+)"

result = ""


def get_local_paths(folder):

    path = list(folder.keys())[0]
    
    instruction = folder[path]
    path = os.path.join("data", path)
    
    dataset = EADataset(path, device="cpu")
    
    local_paths = dataset.video_paths
    if instruction:
        result = re.search(extract_instruction_reg, instruction)
        option = str (result.group(1))
        num = int (result.group(2))

        if option == "first":
            local_paths = local_paths[:num]
        elif option == "last":
            local_paths = local_paths[-num:]
    return local_paths

def read_datasets(path):

    stream = open(path, 'r')
    dictionaries = yaml.load_all(stream).__next__()
    for index, d in enumerate(dictionaries):
        train_dataset = test_dataset = None
        slice = False
        common = False
        train_epochs = None

        if 'name' not in d:
            raise Exception("Name of configuration #{index} not found")

        name = d['name']
        
        if 'data_aug' in d:
            aug_arr = d['data_aug']
            aug_dict = reduce(lambda r, d: r.update(d) or r, aug_arr, {})
            slice = aug_dict['slice']
            common = aug_dict['common']

        train_paths = []
        if 'train' in d:
            for folder in d["train"]:
                train_paths+= get_local_paths(folder)

            train_dataset = EADataset(None, "cuda", video_paths=train_paths, do_slice=slice, do_common=common)
            print(train_paths)
            print(len(train_paths))

            if 'epochs' in d:
                train_epochs = d['epochs']
            
        test_paths = []
        if 'test' in d:
            for folder in d["test"]:
                
                test_paths+= get_local_paths(folder)

            test_dataset = EADataset(None, "cpu", video_paths=test_paths)

        

        yield (name, train_dataset, train_epochs, test_dataset)
         
    

