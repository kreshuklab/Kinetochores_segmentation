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

def ds_creator(data_dir, label_dir):
    """ds_creator function"""

    data_dir = os.listdir(data_dir)
    label_dir = os.listdir(label_dir)

    try:
        assert isinstance(data_dir, list)
    except:
        raise AssertionError('Provide a list of files')
        exit()

    for egfp_data, label in zip(data_dir, label_dir):
        
        # data
        egfp_file = h5py.File(egfp_data, 'r')
        egfp_array = egfp_file['exported_data']

        # Make dims -> C,D,H,W
        egfp_array = np.transpose(egfp_array, (3, 0, 1, 2)).astype('float32')

        # Channel dim required for BCEDigitLoss
        # Final dim -> (1, 48, 128, 128)
        #egfp_array_squeezed = egfp_array[0]
        #egfp_array_squeezed = egfp_array_squeezed[12:60, 436:564, 436:564]

        egfp_array = egfp_array[:, 12:60, 426:554, 426:554]

        # labels
        with open(label, 'rb') as hn:
            data = pickle.load(hn)

        newdata = {}

        # Hard-coded values based on
        # min, max for signals across
        # X, Y, Z
        for k, val in data.items():
            val_list = []
            val_list.append(round(val[0] - 426, 2))
            val_list.append(round(val[1] - 426, 2))
            val_list.append(round(val[2] - 12, 2))
            newdata[k] = val_list

        # Get empty volume and set values for signals
        temp_label = np.ones(shape=(1, 48, 128, 128))

        for keys, vals in newdata.items():
            xval, yval, zval = int(vals[0]), int(vals[1]), int(vals[2])
            temp_label[zval, yval, xval] = 2.0

        
        # save as dataset to be passed to model
        fname = 'EGFP_ds' + label.split('/')[-1].split('.')[0][4:] + '.h5'
        h5file = h5py.File(fname, 'w')
        h5file.create_dataset('raw', data=egfp_array)
        h5file.create_dataset('label', data=temp_label, dtype='f4')
        h5file.close()

if __name__ == '__main__':

    args = parser.parse_args()
    ds_creator(args.datadir_path, args.labeldir_path)
