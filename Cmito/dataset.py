# Import necessary packages
import numpy as np
import h5py
import random
import itertools
from collections import defaultdict
from multiprocessing import Process, Queue
import os
import random


def h5read(filename):
	
	print("Loading " + filename + "...")

	f = h5py.File(filename, "r")
	img = f["main"][()]
	f.close()

	print("Loading complete!")

	return img


def h5write(filename, img):

	f = h5py.File(filename, "w")
	dset = f.create_dataset("main", data=img)
	f.close()
	print("Complete!")


class Dataset():

	def __init__(self, directory, d, loaded=False):
		
		self.directory = directory

		if loaded:
			for (label, data) in d.items():
				setattr(self, label, prep(label, data))

		else:
			for (label, name) in d.items():
				setattr(self, label, prep(label, h5read(os.path.join(directory, name))))


class MultiDataset():

	def __init__(self, directories, d):
		
		self.n = len(directories)
		self.directories = directories
		
		for (label, name) in d.items():
			setattr(self, label, [prep(label, h5read(os.path.join(directory, name))) for directory in directories])


def prep(dtype, data):
	
	if dtype in ["image", "mask"]:
		img = autopad(data.astype(np.float32))
		
		if img.max() > 10:
			return img/255

		else:
			return img


def autopad(img):
    
	if len(img.shape) == 3:
		return np.reshape(img, (1,)+img.shape)

	elif len(img.shape) == 4:
		return np.reshape(img, img.shape)

	else:
		raise Exception("Autopad not applicable.")