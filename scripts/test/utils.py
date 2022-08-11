from __future__ import print_function
import imp
import os

import numpy as np

import torch
from torch.nn.parallel import data_parallel
from torch.cuda import *

from test.model import Model
from test.data import Data


def load_model(opt):
    # Create a model.
    net = opt.net
    net.cuda()
    model = Model(net)

    if opt.chkpt_num > 0:
        model = load_chkpt(model, opt.model_dir, opt.chkpt_num)

    return model.eval()


def load_data(dataset, opt):

    data_loader = Data(dataset, opt, is_train=True)

    return data_loader


def load_chkpt(model, fpath, chkpt_num):
    print("LOAD CHECKPOINT: {} iters.".format(chkpt_num))
    fname = os.path.join(fpath, "model{}.chkpt".format(chkpt_num))
    model.load(fname)

    return model


def save_chkpt(model, fpath, chkpt_num):
    print("SAVE CHECKPOINT: {} iters.".format(chkpt_num))
    fname = os.path.join(fpath, "model{}.chkpt".format(chkpt_num))
    model.save(fname)


def forward(model, sample):
    # Forward pass    
    preds = model(sample)

    return preds
