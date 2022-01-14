"""
Cmito inference.
"""


import time
import argparse
from sys import argv

import numpy as np
import torch

from dataset import *

from test.model import Model
from test.utils import *

from nets.mito_net import *


def em_detector(opt):

	# Output
	fold_out = np.zeros(opt.patch_size + (opt.n_test,), dtype='uint8')

	# Load model
	model = load_model(opt)

	# Load data
	test_loader = load_data(opt.test_data, opt)

	for i in range(opt.n_test):

		t0 = time.time()

		sample = test_loader()
		pred = forward(model, sample)

		mask = pred["mask"].cpu().detach().numpy()
		fold_out[:,:,i] = (mask*255).astype('uint8')


		# Stats
		elapsed = np.round(time.time() - t0, 3)

		if (i+1) % 50 == 0 or (i+1) <=10:
			print("Iter:  " + str(i+1) + ", elapsed time = " + str(elapsed))

	h5write(opt.fwd_dir + opt.output_file, fold_out)



if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--exp_dir", required=True, type=str,
		help="Model path")
	parser.add_argument("--chkpt_num", required=True, type=int,
		help="Model checkpoint number")
	parser.add_argument("--input_file", required=True, type=str,
		help="Input file to detect folds")
	parser.add_argument("--output_file", required=True, type=str,
		help="Output filename")

	opt = parser.parse_args()

	data_dir = ""
	TEST = Dataset(os.path.expanduser(data_dir),
	        {
	            "image": opt.input_file
	        }
	)

	opt.model_dir = opt.exp_dir +'model/'
	opt.fwd_dir = opt.exp_dir + 'forward/'
	opt.exp_name = 'EM detector inference'

	opt.test_data = TEST
	opt.mip = 0
	opt.patch_size = opt.test_data.image.shape[1:3]
	opt.n_test = opt.test_data.image.shape[-1]	

	opt.net = UNet()

	opt.in_spec = ["image"]
	opt.out_spec = ["mask"]

	# GPUs
	opt.gpu_ids = ["0"]
	os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(opt.gpu_ids)

	# Make directories.
	if not os.path.isdir(opt.fwd_dir):
	    os.makedirs(opt.fwd_dir)

	# Run inference.
	print("Running inference: {}".format(opt.exp_name))
	em_detector(opt)
