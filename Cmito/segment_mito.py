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
def segment_mito(pred):

	# Threshold to make binary image
	pthr = 0.5*255 
	pred_thr = (pred>pthr).astype("uint8")

	# Filter dust and segment
	sthr = 5

	pred_cc = cc3d.connected_components(pred_denoise, connectivity=26)

	segid_list, seg_size = np.unique(pred_cc, return_counts=True)
	segid_list = segid_list[1:]; seg_size = seg_size[1:]

	segid_exc = segid_list[seg_size<sthr]
	valid = np.isin(pred_cc, segid_exc)
	pred_cc[valid] = 0


	return pred_cc


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--mito", required=True, type=str,
		help="Path to mito prediction image in cloudvolume, png, or tif format")
	parser.add_argument("--output", required=False, type=str, default="mito_segment.tif",
		help="Path to save mito segmentation image in cloudvolume, png, or tif format")


	args = parser.parse_args()

	pred_path = args.mito
	output_path = args.output

	mito_pred = read_image(pred_path)

	# Segment
	mito_seg = segment_mito(mito_pred)

	# Save output
	save_image(output_path, mito_seg)