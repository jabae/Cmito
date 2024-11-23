import numpy as np
import tifffile as tif
import h5py
import matplotlib.image as image
from cloudvolume import CloudVolume
import itertools
import operator

from model import Model
from nets.unet import UNet

# Read image
def read_image(img_path):

	if img_path[-3:] == "png":

		img = image.imread(img_path)
		img = (img*255).astype("uint8")

	elif img_path[-3:] == "tif":

		img = tif.imread(img_path)

	elif img_path[:3] == "gs:":

		vol = CloudVolume(img_path, parallel=True, progress=False)
		img = vol[:,:,:][...,0]

	elif img_path[-2:] == "h5":

		f = h5py.File(img_path, "r")
		img = f["main"][()]
		f.close()

	return img

# Save image
def save_image(img_path, img):

	if img_path[-3:] == "png":

		if len(img.shape)==2:
			img = np.concatenate(([img.reshape(img.shape+(1,))]*3), axis=2)
			
		image.imsave(img_path, img)

	elif img_path[-3:] == "tif":

		tif.imwrite(img_path, img)

# Load model
def load_model(model_path):

	net = UNet()
	net.cuda()
	model = Model(net)
	
	model = load_chkpt(model, model_path)

	return model.eval()

# Load chkpt file
def load_chkpt(model, path):

  model.load(path)

  return model

# Chunk image
def chunk_bboxes(vol_size, chunk_size, overlap=(0, 0), offset=None, mip=0):

	if mip > 0:
		mip_factor = 2 ** mip
		vol_size = (vol_size[0]//mip_factor,
		            vol_size[1]//mip_factor)

		chunk_size = (chunk_size[0]//mip_factor,
		              chunk_size[1]//mip_factor)

		overlap = (overlap[0]//mip_factor,
		           overlap[1]//mip_factor)

		if offset is not None:
			offset = (offset[0]//mip_factor,
	              offset[1]//mip_factor)

	x_bnds = bounds1D_overlap(vol_size[0], chunk_size[0], overlap[0])
	y_bnds = bounds1D_overlap(vol_size[1], chunk_size[1], overlap[1])

	bboxes = [tuple(zip(xs, ys))
	          for (xs, ys) in itertools.product(x_bnds, y_bnds)]

	if offset is not None:
	  bboxes = [(tuple(map(operator.add, bb[0], offset)),
	             tuple(map(operator.add, bb[1], offset)))
	            for bb in bboxes]

	return bboxes

def bounds1D_overlap(full_width, step_size, overlap=0):

	assert step_size > 0, "invalid step_size: {}".format(step_size)
	assert full_width > 0, "invalid volume_width: {}".format(full_width)
	assert overlap >= 0, "invalid overlap: {}".format(overlap)

	start = 0
	end = step_size

	bounds = []
	while end < full_width:
	  bounds.append((start, end))

	  start += step_size - overlap
	  end = start + step_size

	# last window
	if end>=full_width:
		start = full_width - step_size
		end = full_width
		bounds.append((start, end))
		
	return bounds