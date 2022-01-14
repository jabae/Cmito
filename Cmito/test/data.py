from __future__ import print_function
import imp
import numpy as np
import math

import torch
import torch.utils.data
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn

from time import time


downsample = nn.AvgPool2d(kernel_size=(2,2), stride=(2,2), padding=0)

def worker_init_fn(worker_id):
    # Each worker already has its own random state (Torch).
    seed = torch.IntTensor(1).random_()[0]
    # print("worker ID = {}, seed = {}".format(worker_id, seed))
    np.random.seed(seed)

        
class Dataset(torch.utils.data.Dataset):
    def __init__(self, multidataset, mip):
        super(Dataset, self).__init__()

        self.image = multidataset.image
        
        self.mip = mip
        
        self.size = self.image.shape[3]

    def __len__(self):

        return self.size

    def __getitem__(self, idx):
        image = self.image[:,:,:,idx]
        
        sample = {"image": image}

        sample["image"] = torch.from_numpy(sample["image"].copy())
        for i in range(self.mip):
            sample["image"] = downsample(sample["image"])
            
        return sample


class Data(object):
    def __init__(self, data, opt, is_train=True):
        self.build(data, opt, is_train)

    def __call__(self):
        sample = next(self.dataiter)
        for k in sample:
            is_input = k in self.inputs
            sample[k].requires_grad_(is_input)
            sample[k] = sample[k].cuda(non_blocking=(not is_input))

        return sample

    def requires_grad(self, key):
        return self.is_train and (key in self.inputs)

    def build(self, data, opt, is_train):    

        dataset = Dataset(data, opt.mip)
        
        dataloader = DataLoader(dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=1,
                                pin_memory=True,
                                worker_init_fn=worker_init_fn)

        # Attributes
        self.dataiter = iter(dataloader)
        self.inputs = ['image']
        self.is_train = is_train