"""
Remove dust segments
"""

import argparse
from sys import argv

import numpy as np
import pandas as pd
import tifffile as tif

from cloudvolume import CloudVolume


## Load volume
def load_volume(path, res):
	"""
	INPUT:
	path : Path of the volume
	res : (x, y, z) resolution in nm
	OUTPUT:
	vol : Image volume
	"""

	if path[:2] == "gs":

		cloud_vol = CloudVolume(path, mip=res, 
			parallel=True, progress=False)
		vol = cloud_vol[:,:,:][:,:,:,0]

	else:

		vol = tif.imread(path)

	return vol


## Write volume
def save_volume(path, vol, res):
	"""
	INPUT:
	path : Path of the volume
	vol : Image volume
	res : (x, y, z) resolution in nm
	"""

	if path[:2] == "gs":

		info = CloudVolume.create_new_info(
	  num_channels    = 1,
	  layer_type      = 'segmentation',
	  data_type       = 'uint16', # Channel images might be 'uint8'
	  # raw, jpeg, compressed_segmentation, fpzip, kempressed, compresso
	  encoding        = 'raw', 
	  resolution      = res, # Voxel scaling, units are in nanometers
	  voxel_offset    = [0, 0, 0], # x,y,z offset in voxels from the origin
	  # Pick a convenient size for your underlying chunk representation
	  # Powers of two are recommended, doesn't need to cover image exactly
	  chunk_size      = [ 512, 512, 1 ], # units are voxels
	  volume_size     = vol.shape, # e.g. a cubic millimeter dataset
		)

		cloudvol = CloudVolume(path, parallel=True, progress=True, info=info)
		cloudvol.commit_info()

		cloudvol[:,:,:] = vol

	else:

		vol = tif.imwrite(path, vol)

	return vol


## Remove dust
def remove_dust(seg, size_thr):
	"""
	INPUT:
	seg: Segmentation volume
	size_thr: Size threshold
	OUTPUT:
	seg: Refined segmentation volume
	"""

	nsec = seg.shape[2]
	for i in range(nsec):

		seg_sect = seg[:,:,i]
		seg_sect_list, seg_sect_size = np.unique(seg_sect, return_counts=True)

		seg_sect_new = np.copy(seg_sect)
		exc_seg_list = seg_sect_list[seg_sect_size<size_thr]
		exc_mask = np.isin(seg_sect, exc_seg_list)
		seg_sect_new[exc_mask] = 0

		seg[:,:,i] = seg_sect_new

	return seg


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--mito_seg", required=True, type=str,
		help="Mitochondria segmentation volume")
	parser.add_argument("--size_thr", required=True, type=int,
		help="Size threshold in voxels")
	parser.add_argument("--res", nargs=3, required=True, type=int,
		help="(x, y, z) resolution of the segmentation")
	parser.add_argument("--outpath", required=True, type=str,
		help="Output path")

	opt = parser.parse_args()

	seg_path = opt.mito_seg
	res = opt.res

	# Load volume
	mito_seg = load_volume(seg_path, res)

	# Remove dust
	mito_seg = remove_dust(mito_seg, opt.size_thr)

	# Save volume
	save_vol(opt.outpath, mito_seg, res)