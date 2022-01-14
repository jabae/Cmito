from __future__ import print_function
import imp
import os

import numpy as np

import torch
from torch.nn.parallel import data_parallel
from torch.cuda import *

from train.data import Data
from train.model import Model



def load_model(opt):
    # Create a model.
    net = opt.net
    net.cuda()
    model = Model(net, opt)

    if opt.pretrain:
        print("Loading {}...".format(opt.pretrain))
        model.load(opt.pretrain)
    if opt.chkpt_num > 0:
        model = load_chkpt(model, opt.model_dir, opt.chkpt_num)

    return model.train(), net


def load_chkpt(model, fpath, chkpt_num):
    print("LOAD CHECKPOINT: {} iters.".format(chkpt_num))
    fname = os.path.join(fpath, "model{}.chkpt".format(chkpt_num))
    model.load(fname)
    return model


def save_chkpt(model, fpath, chkpt_num):
    print("SAVE CHECKPOINT: {} iters.".format(chkpt_num))
    fname = os.path.join(fpath, "model{}.chkpt".format(chkpt_num))
    model.save(fname)


def load_data(dataset, aug, opt):

    data_loader = Data(dataset, aug, opt, is_train=True)

    return data_loader


def forward(model, sample, opt):
    # Forward pass
    if len(opt.gpu_ids) > 1:
        
        losses, preds = data_parallel(model, sample)
    else:
        
        losses, preds = model(sample)

    # Average over minibatch
    losses = {k: v.mean() for k, v in losses.items()}

    return losses, preds
