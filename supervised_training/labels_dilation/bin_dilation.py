import h5py
import numpy as np
import torch
from scipy import ndimage
import argparse
import glob

#files = glob.glob('/home/samudre/embldata/EGFP/processed_data/val/*')
#files = glob.glob('/home/samudre/embldata/EGFP/new_processed/data/*')

parser = argparse.ArgumentParser()
parser.add_argument('--dir', default='', help='path to directory with all hdf5 files')
parser.add_argument('--iter', default=1, help='iterations for dilation of labels')

def bin_dilation(datadir, iterations):
	"""
	datadir: directory with all data HDF5 files
	iterations: label dilation parameter
	"""
	files = glob.glob(datadir + '/*')

	for each in files:
		h5file = h5py.File(each, 'r')
		label = h5file['label']
		raw = torch.tensor(h5file['raw'])
		label_mod = torch.tensor(ndimage.binary_dilation(label, iterations=iterations).astype(label.dtype))
		fname = each.split('/')[-1]
		newh5 = h5py.File(fname, 'w')
		newh5.create_dataset('raw', data=raw)
		newh5.create_dataset('label', data=label_mod)
		newh5.close()

if __name__ == '__main__':
	args = parser.parse_args()
	datadir = args.dir
	iterations = args.iter

	bin_dilation(datadir, iterations)