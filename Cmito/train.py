"""
Train CmitoNet.
"""

import os
import time
import argparse

import torch

from train.model import Model

from train.logger import Logger
from train.utils import *

from dataset import *

from nets.mito_net import *


def train(opt):

    model, net = load_model(opt)

    optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)
    print(optimizer)

    # Initial checkpoint
    save_chkpt(model, opt.model_dir, opt.chkpt_num)

    # Training loop
    print("========== BEGIN TRAINING LOOP ==========")
    with Logger(opt) as logger:
        # Log parameters
        logger.log_parameters(vars(opt))
        logger.log_command()

        chkpt_epoch = opt.chkpt_num//int(opt.n_train/opt.batch_size)
        chkpt_iter = opt.chkpt_num%int(opt.n_train/opt.batch_size)

        i = opt.chkpt_num
        for epoch in range(chkpt_epoch, opt.max_epoch):

            # Data loaders (Reset every epoch)
            train_loader = load_data(opt.train_data, opt.train_augment, opt)
            val_loader = load_data(opt.val_data, opt.val_augment, opt)


            for it in range(chkpt_iter, int(opt.n_train/opt.batch_size)):
                # Timer
                t0 = time.time()
        
                # Load Training samples.
                sample = train_loader()

                # Optimizer step
                optimizer.zero_grad()
                losses, preds = forward(model, sample, opt)
                total_loss = sum(losses[k] for k in opt.out_spec)
                losses["all"] = total_loss/len(opt.out_spec)
                total_loss.backward()
                optimizer.step()

                # Elapsed time
                elapsed = time.time() - t0
                
                # Record keeping
                logger.record('train', losses, elapsed=elapsed)

                # Log & display averaged stats
                if (i+1) % opt.avgs_intv == 0 or i < opt.warm_up:
                    logger.check('train', i+1)

                # Logging images
                if (i+1) % opt.imgs_intv == 0:
                    logger.log_images('train', i+1, preds, sample)

                # Evaluation loop
                if (i+1) % opt.eval_intv == 0:
                    eval_loop(i+1, model, val_loader, opt, logger)
                    val_loader = load_data(opt.val_data, opt.val_augment, opt)

                # Model checkpoint
                if (i+1) % opt.chkpt_intv == 0:
                    save_chkpt(model, opt.model_dir, i+1)

                # Reset timer.
                t0 = time.time()

                i = i + 1
                

def eval_loop(iter_num, model, data_loader, opt, logger):
    if not opt.no_eval:
        model.eval()

    # Evaluation loop
    print("---------- BEGIN EVALUATION LOOP ----------")
    with torch.no_grad():
        t0 = time.time()
        for i in range(opt.eval_iter):
            sample = data_loader()
            losses, preds = forward(model, sample, opt)
            losses["all"] = sum(losses[k] for k in opt.out_spec)/len(opt.out_spec)
            elapsed = time.time() - t0

            # Record keeping
            logger.record('test', losses, elapsed=elapsed)

            # Restart timer.
            t0 = time.time()

    # Log & display averaged stats.
    logger.check('test', iter_num)
    print("-------------------------------------------")

    model.train()


if __name__ == "__main__":

		# Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", required=True, type=str,
        help="Train path")
    parser.add_argument("--train_image", required=True, type=str,
        help="Train image data in h5")
    parser.add_argument("--train_label", required=True, type=str,
        help="Train mask data in h5")
    parser.add_argument("--val_image", required=True, type=str,
        help="Validation image data in h5")
    parser.add_argument("--val_label", required=True, type=str,
        help="Validation mask data in h5")
    parser.add_argument("--chkpt_num", required=True, type=int,
        help="Model checkpoint number to start training")
    parser.add_argument("--pretrain", required=False, type=str, default=None,
    		help="Pretrained weights (if any)")
    parser.add_argument("--max_epoch", required=False, type=int, default=2000,
    		help="Number of epochs")


    opt = parser.parse_args()

    data_dir = ""
    TRAIN = Dataset(os.path.expanduser(data_dir),
            {
                "image": opt.train_image,
                "mask": opt.train_label
            }
    )

    VAL = Dataset(os.path.expanduser(data_dir),
            {
                "image": opt.val_image,
                "mask": opt.val_label
            }
    )

    opt.log_dir = opt.exp_dir + 'log/'    
    opt.model_dir = opt.exp_dir + 'model/'
    opt.exp_name = 'EM detector'

    opt.train_data = TRAIN
    opt.val_data = VAL
    opt.mip = 0
    opt.n_train = opt.train_data.image.shape[-1]

    opt.gpu_ids = ["0","1","2","3","4","5","6","7"]

    opt.batch_size = len(opt.gpu_ids)
    opt.num_workers = len(opt.gpu_ids)

    opt.net = UNet()
    
    opt.max_epoch = opt.max_epoch
    opt.chkpt_intv = 2000
    opt.avgs_intv = 100 
    opt.imgs_intv = 500
    opt.warm_up = 100
    
    opt.eval_iter = 15
    opt.eval_intv = 1000
    opt.no_eval = True

    opt.in_spec = ['image']
    opt.out_spec = ['mask']
    opt.train_augment = ['flip','rotate90','contrast']
    opt.val_augment = []
    opt.pretrain = opt.pretrain

    opt.lr = 0.0005

    # GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(opt.gpu_ids)


    # Make directories.
    if not os.path.isdir(opt.exp_dir):
        os.makedirs(opt.exp_dir)
    if not os.path.isdir(opt.log_dir):
        os.makedirs(opt.log_dir)
    if not os.path.isdir(opt.model_dir):
        os.makedirs(opt.model_dir)

    # Run experiment.
    print("Running experiment: {}".format(opt.exp_name))
    train(opt)
