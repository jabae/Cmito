import numpy as np


def block_augment(sample):
	"""Performs contrast/brightness augmentation on img.

	Args:
		sample: Dictionary of (np array: <ch,z,x,y>) image and mask
	"""

	l = sample["image"].shape[1]

	if np.random.uniform() < 0.2:

		r = np.random.rand()
		xloc = np.random.randint(l//2-50,l//2+50)
		yloc = np.random.randint(l//2-50,l//2+50)

		sample["image"][:,:xloc,:yloc] = sample["image"][:,:xloc,:yloc] - (np.random.rand() - 0.5)*0.5
		sample["image"][:,:xloc,yloc:] = sample["image"][:,:xloc,yloc:] - (np.random.rand() - 0.5)*0.5
		sample["image"][:,xloc:,yloc:] = sample["image"][:,xloc:,yloc:] - (np.random.rand() - 0.5)*0.5
		sample["image"][:,xloc:,:yloc] = sample["image"][:,xloc:,:yloc] - (np.random.rand() - 0.5)*0.5

		sample["image"] = np.clip(sample["image"], 0, 1)


	return sample