"""N channel data creator"""

import argparse
import h5py
import numpy as np
from scipy import ndimage



parser = argparse.ArgumentParser()
parser.add_argument('egfp_dir', default='', help='Path to EGFP files directory')
parser.add_argument('mcherry_dir', default='', help='Path to mcherry files directory')
parser.add_argument('n_channel', default='', help='number of channels for the vol')


# prefix_mc = '/home/samudre/embldata/mcherry/dt/'
# prefix_egfp = '/home/samudre/embldata/EGFP/new_processed/data/'


class NchannelCreator:

    """NChannel_creator class"""

    def __init__(self):
        pass

    @staticmethod
    def label_transform(ground_truth):
        """
        ground_truth: labels to be transformed

        """
        inv_label = np.logical_not(ground_truth)

        # Take distance transform
        label_dist_transform = ndimage.distance_transform_edt(inv_label)

        # A spherical label of radius 3
        label_thresh_tr = label_dist_transform > 3

        # Revert the label back to original for foreground/ background
        label_thresh_tr = np.logical_not(label_thresh_tr).astype(np.float64)

        return label_thresh_tr

    def raw_formatter(self, egfp_dir, mcherry_dir, n_channel):
        """
        egfp_dir: path/ prefix for egfp data
        mcherry_dir: path/prefix for mcherry data
        n_channel: num of channels in the vol

        """
        if n_channel == 2:
            for filenum in range(18):
                egfp_file = h5py.File(egfp_dir + 'EGFP_' + str(filenum) + '_new.h5', 'r')
                mcherry_file = h5py.File(mcherry_dir + 'mcherry_ds' + str(filenum) + '.h5', 'r')

                # load raw datasets
                egfp_raw = egfp_file['raw']
                mcherry_raw = mcherry_file['raw']

                # load the annotation
                ground_truth = egfp_file['label']

                label_thresh_tr = self.label_transform(ground_truth)

                # Concatenate the raw files for 2 channel input and normalize
                concat_raw = np.concatenate((egfp_raw, mcherry_raw), axis=0)
                concat_raw_norm = ((concat_raw - np.min(concat_raw)) / (np.max(concat_raw) - np.min(concat_raw)))

                # Set the output file name
                outfile_name = '2ch_spherical' + str(filenum) + '.h5'

                channel_2_file = h5py.File(outfile_name, 'w')
                channel_2_file.create_dataset('raw', data=concat_raw_norm)
                channel_2_file.create_dataset('label', data=label_thresh_tr)
                channel_2_file.close()

        elif n_channel == 6:
            for filenum in range(0, 18, 3):
                egfp_file1 = h5py.File(egfp_dir + 'EGFP_' + str(filenum) + '_new.h5', 'r')
                egfp_file2 = h5py.File(egfp_dir + 'EGFP_' + str(filenum+1) + '_new.h5', 'r')
                egfp_file3 = h5py.File(egfp_dir + 'EGFP_' + str(filenum+2) + '_new.h5', 'r')

                mcherry_file1 = h5py.File(mcherry_dir + 'mcherry_ds' + str(filenum) + '.h5', 'r')
                mcherry_file2 = h5py.File(mcherry_dir + 'mcherry_ds' + str(filenum+1) + '.h5', 'r')
                mcherry_file3 = h5py.File(mcherry_dir + 'mcherry_ds' + str(filenum+2) + '.h5', 'r')

                egfp_file1_raw = egfp_file1['raw']
                egfp_file2_raw = egfp_file2['raw']
                egfp_file3_raw = egfp_file3['raw']

                mcherry_file1_raw = mcherry_file1['raw']
                mcherry_file2_raw = mcherry_file2['raw']
                mcherry_file3_raw = mcherry_file3['raw']

                concat_raw = np.concatenate((egfp_file1_raw, mcherry_file1_raw, egfp_file2_raw,\
                                    mcherry_file2_raw, egfp_file3_raw, mcherry_file3_raw), axis=0)

                concat_raw_norm = ((concat_raw - np.min(concat_raw)) / (np.max(concat_raw) - np.min(concat_raw)))

                ground_truth = egfp_file2['label']

                label_thresh_tr = self.label_transform(ground_truth)

                outfile_name = '6ch_spherical' + str(filenum) + '.h5'

                channel_6_file = h5py.File(outfile_name, 'w')
                channel_6_file.create_dataset('raw', data=concat_raw_norm)
                channel_6_file.create_dataset('label', data=label_thresh_tr)
                channel_6_file.close()

if __name__ == '__main__':
    args = parser.parse_args()
    egfp_dir = args.egfp_dir
    mcherry_dir = args.mcherry_dir
    n_channel = args.n_channel

    vol_creator = NchannelCreator()
    vol_creator.raw_formatter(egfp_dir, mcherry_dir, n_channel)
