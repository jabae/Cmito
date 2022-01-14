import numpy as np


def darkline_augment(sample):
	"""Performs contrast/brightness augmentation on img.

	Args:
		sample: Dictionary of (np array: <ch,z,x,y>) image and mask
	"""

	l = sample["image"].shape[1]
	w = np.random.randint(10,20)

	if np.random.uniform() < 0.2:

		r = np.random.rand()
		loc = np.random.randint(w+5,l-(w+5))
		b = np.abs(np.random.rand() - 0.5)*0.5

		if r < 0.5: 
			sample["image"][:,:,loc:loc+w] = sample["image"][:,:,loc:loc+w] - b
		else:
			sample["image"][:,loc:loc+w,:] = sample["image"][:,loc:loc+w,:] - b

		sample["image"] = np.clip(sample["image"], 0, 1)


	return sample