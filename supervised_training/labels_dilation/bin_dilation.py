import h5py
import numpy as np
import torch
from scipy import ndimage
import glob

files = glob.glob('/home/samudre/embldata/EGFP/processed_data/val/*')

for each in files:
	h5file = h5py.File(each, 'r')
	label = h5file['label']
	raw = torch.tensor(h5file['raw'])
	label_mod = torch.tensor(ndimage.binary_dilation(label).astype(label.dtype))
	fname = each.split('/')[-1]
	newh5 = h5py.File(fname, 'w')
	newh5.create_dataset('raw', data=raw)
	newh5.create_dataset('label', data=label_mod)
	newh5.close()
