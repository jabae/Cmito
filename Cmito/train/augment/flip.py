import numpy as np


def flip_augment(sample):
  """Performs flip augmentation on img.

  Args:
    sample: Dictionary of (np array: <ch,z,x,y>) image and mask
  """

  # z flip
  if np.random.rand() < 0.5:
    for k in sample.keys():
      sample[k] = np.flip(sample[k], axis=1)

  # x flip
  if np.random.rand() < 0.5:
    for k in sample.keys():
      sample[k] = np.flip(sample[k], axis=2)


  return sample