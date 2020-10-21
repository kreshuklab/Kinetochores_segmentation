"""Final input dataset creator"""

import pickle
import argparse
import numpy as np
import h5py

# egfiles = glob.glob('~/embldata/EGFP/datafiles/*')
# labelfiles = glob.glob('~/embldata/EGFP/labeldir/*')

parser = argparse.ArgumentParser()
parser.add_argument('datadir_path', default='', help='Path to data files')
parser.add_argument('labeldir_path', default='', help='Path to label files')
parser.add_argument('data_channel', default='', help='Provide EGFP or mcherry')

def ds_creator(data_dir, label_dir, data_channel):
    """
    data_dir: path to raw data files
    label_dir: path to gt label files
    data_channel: raw data channel (EGFP/ mcherry)
    """

    data_dir = os.listdir(data_dir)
    label_dir = os.listdir(label_dir)

    for raw_data, label in zip(data_dir, label_dir):
        
        # data
        raw_vol = h5py.File(raw_data, 'r')['exported_data']

        # Make dims -> C,D,H,W
        raw_vol = np.transpose(raw_vol, (3, 0, 1, 2)).astype('float32')

        # Channel dim required for BCEDigitLoss
        # Final dim -> (1, 48, 128, 128)
        #raw_vol_squeezed = raw_vol[0]
        #raw_vol_squeezed = raw_vol_squeezed[12:60, 436:564, 436:564]

        raw_vol = raw_vol[:, 12:60, 426:554, 426:554]

        # labels
        with open(label, 'rb') as hn:
            data = pickle.load(hn)

        try:
            assert isinstance(data, dict)
        except:
            raise AssertionError('Input pickle should be a dict.')
            exit()

        newdata = {}

        # Hard-coded values based on
        # min, max for signals across
        # X, Y, Z
        for key, val in data.items():
            val_list = []
            val_list.append(round(val[0] - 426, 2))
            val_list.append(round(val[1] - 426, 2))
            val_list.append(round(val[2] - 12, 2))
            newdata[key] = val_list

        # Get empty volume and set values for signals
        temp_label = np.zeros(shape=(1, 48, 128, 128))

        for keys, vals in newdata.items():
            xval, yval, zval = int(vals[0]), int(vals[1]), int(vals[2])
            temp_label[zval, yval, xval] = 1.0

        
        # save as dataset to be passed to model
        fname = data_channel + label.split('/')[-1].split('.')[0][4:] + '.h5'
        h5file = h5py.File(fname, 'w')
        h5file.create_dataset('raw', data=raw_vol)
        h5file.create_dataset('label', data=temp_label, dtype='f4')
        h5file.close()

if __name__ == '__main__':

    args = parser.parse_args()
    ds_creator(args.datadir_path, args.labeldir_path)
