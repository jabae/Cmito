import numpy as np


def contrast_augment(sample):
	"""Performs contrast/brightness augmentation on img.

	Args:
		sample: Dictionary of (np array: <ch,z,x,y>) image and mask
	"""

	f_s = 1
	f_b = 0.5

	a = 1 + (np.random.rand() - 0.5) * f_s
	b = (np.random.rand() - 0.5) * f_b
	g = (np.random.rand()*2 - 1)

	sample["image"] = sample["image"] * a + b
	sample["image"] = np.clip(sample["image"], 0, 1)
	sample["image"] = sample["image"]**(2**g)


	return sample