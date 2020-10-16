import h5py
import tifffile
import numpy as np
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--dir', default='', help='path to directory with all hdf5 files')

def hdf5_to_tiff(hdf5_dir):
	hdf5list = glob.glob(hdf5_dir + '*')
	#print(hdf5list)
	for each in hdf5list:
		#tiff_name = each[:-3] + '.tif'
		file = h5py.File(each, 'r') # read the file
		tiff_name = 'ds' + each.split('/')[-1].split('.')[0].split('_')[1] + '.tif'

		# providing (48,128,128) for the tif as required by stardist
		data = file['label'][0] # modify this as per the dataset name in the hdf5 file
		tifffile.imwrite(tiff_name, data) # write the tiff file

if __name__ == '__main__':
	args = parser.parse_args()
	hdf5_dir = args.dir

	hdf5_to_tiff(hdf5_dir)