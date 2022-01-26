"""
Run inference on volume locally
"""

import time
import argparse

import numpy as np
import torch

from cloudvolume import CloudVolume

from test.model import Model
from test.utils import *

from dataset import *
from utils import *

from nets.mito_net import *


def detect(opt):

	# Output
	detect_out = np.zeros(opt.patch_size + [opt.n_test,], dtype='uint8')

	# Load model
	model = load_model(opt)

	# Load data
	test_loader = load_data(opt.test_data, opt)

	for i in range(opt.n_test):

		t0 = time.time()

		sample = test_loader()
		pred = forward(model, sample)

		mask = pred["mask"].cpu().detach().numpy()
		detect_out[:,:,i] = (mask*255).astype('uint8')

		# Stats
		elapsed = np.round(time.time() - t0, 3)


	return detect_out


def merge(img_stack, bbox_list, sect_sz, ovlap_sz):

	img_sect = np.zeros(sect_sz, dtype="uint8")

	for i in range(len(bbox_list)):

		img_patch = img_stack[ovlap_sz[0]//2:-ovlap_sz[0]//2,
										ovlap_sz[1]//2:-ovlap_sz[1]//2,i]

		b = bbox_list[i]

		img_patch_orig = img_sect[b[0][0]+ovlap_sz[0]//2:b[1][0]-ovlap_sz[0]//2,
															b[0][1]+ovlap_sz[1]//2:b[1][1]-ovlap_sz[1]//2]
		img_patch_new = img_patch*(img_patch>=img_patch_orig) + img_patch_orig*(img_patch<img_patch_orig)
		
		img_sect[b[0][0]+ovlap_sz[0]//2:b[1][0]-ovlap_sz[0]//2,
						b[0][1]+ovlap_sz[1]//2:b[1][1]-ovlap_sz[1]//2] = img_patch_new


	return img_sect



if __name__ == "__main__":

	# Arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--exp_dir", required=True, type=str,
		help="Model path")
	parser.add_argument("--chkpt_num", required=True, type=int,
		help="Model checkpoint number")
	parser.add_argument("--src_path", required=True, type=str,
		help="Cloud path of input volume")
	parser.add_argument("--dst_path", required=True, type=str,
		help="Cloud path of output volume")
	parser.add_argument("--bbox_start", nargs=3, type=int,
		help="Bounding box start coordinates of volume")
	parser.add_argument("--bbox_end", nargs=3, type=int,
		help="Bounding box end coordinates of volume")
	parser.add_argument("--patch_size", nargs=2, required=True, type=int,
		help="Input patch size")
	parser.add_argument("--overlap_size", nargs=2, required=True, type=int,
		help="Overlap size")
	parser.add_argument("--mip", required=True, type=int,
		help="Mip level")
	parser.add_argument("--res", nargs=3, type=int,
		help="Resolution of output volume")


	opt = parser.parse_args()

	mip = opt.mip
	res = opt.res

	# Model directory
	opt.model_dir = opt.exp_dir + 'model/'
	opt.net = UNet()

	opt.in_spec = ["image"]
	opt.out_spec = ["mask"]

	# GPUs
	opt.gpu_ids = ["9"]
	os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(opt.gpu_ids)

	# Sizes
	bbox_st = opt.bbox_start
	bbox_end = opt.bbox_end

	nsect = bbox_end[2]-bbox_st[2]
	img_sz = (bbox_end[0]-bbox_st[0], bbox_end[1]-bbox_st[1])
	patch_sz = tuple(opt.patch_size)
	ovlap_sz = tuple(opt.overlap_size)
	
	# Cloud volumes
	# Input
	src_path = opt.src_path
	src_vol = CloudVolume(src_path, mip=mip, parallel=True, progress=False)
	print(">>> Input volume loaded.")

	# Output
	dst_path = opt.dst_path
	info = CloudVolume.create_new_info(
		num_channels    = 1,
		layer_type      = 'image',
		data_type       = 'uint8',
		encoding        = 'raw', 
		resolution      = [res[0], res[1], res[2]], 
		voxel_offset    = [bbox_st[0], bbox_st[1], bbox_st[2]], 
		chunk_size      = [ 256, 256, 1 ], # units are voxels
		volume_size     = [ img_sz[0], img_sz[1], nsect ], # e.g. a cubic millimeter dataset
	)

	dst_vol = CloudVolume(dst_path, parallel=True, progress=False, info=info)
	dst_vol.commit_info()
	print(">>> Output volume loaded.")

	bbox_list = chunk_bboxes_2d(img_sz, patch_sz, ovlap_sz)

	zoff = bbox_st[2]
	for i in range(nsect):

		zidx = zoff + i

		img = src_vol[bbox_st[0]:bbox_end[0],bbox_st[1]:bbox_end[1],zidx]

		patch_list = []
		for b in bbox_list:

			img_patch = img[b[0][0]:b[1][0],b[0][1]:b[1][1]]
			patch_list.append(img_patch.reshape(patch_sz+(1,)))

		img_stack = np.concatenate(patch_list, axis=2)
		
		data_dir = "" 
		TEST = Dataset(os.path.expanduser(data_dir),
						{
							"image": img_stack
						},
						loaded=True
		)
		opt.test_data = TEST
		opt.n_test = len(patch_list)

		# Run inference
		pred_stack = detect(opt)

		pred_sect = merge(pred_stack, bbox_list, img_sz, ovlap_sz)
		dst_vol[:,:,zidx] = pred_sect.reshape(img_sz+(1,))


		print("{} / {} sections complete!".format(i+1, nsect))
