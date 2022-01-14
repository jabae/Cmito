from __future__ import print_function
import imp
import numpy as np
import math

import torch
import torch.utils.data
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn

from train.augment.augmentor import Augmentor

from time import time


downsample_avg = nn.AvgPool2d(kernel_size=(2,2), stride=(2,2), padding=0)
downsample_max = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), padding=0)

def worker_init_fn(worker_id):
    # Each worker already has its own random state (Torch).
    seed = torch.IntTensor(1).random_()[0]
    np.random.seed(seed)

        
class Dataset(torch.utils.data.Dataset):
    def __init__(self, multidataset, aug_params, mip):
        super(Dataset, self).__init__()

        self.image = multidataset.image
        self.mask = multidataset.mask 

        self.mip = mip
        
        self.size = self.image.shape[3]

        augmentor = Augmentor(aug_params)
        self.augmentor = augmentor

    def __len__(self):

        return self.size

    def __getitem__(self, idx):
        image = self.image[:,:,:,idx]
        mask = self.mask[:,:,:,idx]
        
        sample = {"image": image, "mask": mask}

        # Augmentation
        sample = self.augmentor(sample)
        for k in sample.keys():
            sample[k] = torch.from_numpy(sample[k].copy())

        # for i in range(self.mip):
        #     sample["image"] = downsample_avg(sample["image"])
        #     sample["mask"] = downsample_max(sample["mask"])

        return sample


class Data(object):
    def __init__(self, data, aug, opt, is_train=True):
        self.build(data, aug, opt, is_train)

    def __call__(self):
        sample = next(self.dataiter)
        for k in sample:
            is_input = k in self.inputs
            sample[k].requires_grad_(is_input)
            sample[k] = sample[k].cuda(non_blocking=(not is_input))

        return sample

    def requires_grad(self, key):
        return self.is_train and (key in self.inputs)

    def build(self, data, aug, opt, is_train):

        aug_params = {'flip': False, 'rotate90': False, 'contrast': False, 'blackpad': False, 'darkline': False, 'block': False}
        for k in aug:
            aug_params[k] = True

        dataset = Dataset(data, aug_params, opt.mip)       

        dataloader = DataLoader(dataset,
                                batch_size=opt.batch_size,
                                shuffle=True,
                                num_workers=opt.num_workers,
                                pin_memory=True,
                                worker_init_fn=worker_init_fn)

        # Attributes
        self.dataiter = iter(dataloader)
        self.inputs = ['image']
        self.is_train = is_train

