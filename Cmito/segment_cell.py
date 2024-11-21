"""
Script to segment cell from predicted nuclei.
"""

import numpy as np
import matplotlib.pyplot as plt
import cc3d
from scipy.spatial import cKDTree
import argparse

from utils import read_image, save_image



# Segment cell based on predicted nuclei
def segment_cell(img, pred):

	# Threshold to make binary image
	pthr = 0.9*255 
	pred_thr = (pred>pthr).astype("uint8")

	# Denoise
	kw = 10
	dthr = 0.6
	h = img.shape[0]
	w = img.shape[1]

	kernel = np.ones((kw,kw), dtype="uint8")

	pred_denoise = np.copy(pred_thr)
	for i in range(h-kw):
		for j in range(w-kw):

			if np.mean(img[i:i+kw,j:j+kw])<dthr:
				pred_denoise[i+kw//2,j+kw//2] = 0

	# Filter dust
	sthr = 10

	pred_cc = cc3d.connected_components(pred_denoise, connectivity=8)

	segid_list, seg_size = np.unique(pred_cc, return_counts=True)
	segid_list = segid_list[1:]; seg_size = seg_size[1:]

	segid_exc = segid_list[seg_size<sthr]
	valid = np.isin(pred_cc, segid_exc)
	pred_cc[valid] = 0

	# Image mask
	kw = 15
	ithr = 40

	kernel = np.ones((kw,kw), dtype="uint8")

	img_mask = np.zeros((h,w), dtype="uint8")
	for i in range(h-kw):
		for j in range(w-kw):

			if np.mean(img[i:i+kw,j:j+kw])<ithr:
				img_mask[i+kw//2,j+kw//2] = 1

	# Segment cells
	ccstats = cc3d.statistics(pred_cc)

	cent_list = ccstats["centroids"]
	segid_list = np.arange(cent_list.shape[0])
	cent_list = cent_list[1:]; segid_list = segid_list[1:]

	valid = ~np.isnan(cent_list[:,0])
	cent_list = cent_list[valid,:]
	segid_list = segid_list[valid]

	idx_all = np.where(img_mask==0)
	idx_all = np.array(idx_all).T

	tree = cKDTree(cent_list)
	dd, ii = tree.query(idx_all, k=1)

	cell_seg = np.zeros((h,w), dtype="uint16")
	for i in range(idx_all.shape[0]):

		cell_seg[idx_all[i,0], idx_all[i,1]] = segid_list[ii[i]]


	return cell_seg


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--nucleus", required=True, type=str,
		help="Path to nuecleus prediction image in cloudvolume, png, or tif format")
	parser.add_argument("--image", required=True, type=str,
		help="Path to image")
	parser.add_argument("--output", required=False, type=str, default="cell_segment.tif",
		help="Path to save cell segmentation image in cloudvolume, png, or tif format")


	args = parser.parse_args()

	img_path = args.image
	pred_path = args.nucleus
	output_path = args.output

	img = read_image(img_path)
	nuc_pred = read_image(pred_path)

	# Segment
	cell_seg = segment_cell(img, nuc_pred)

	# Save output
	save_image(output_path, cell_seg)