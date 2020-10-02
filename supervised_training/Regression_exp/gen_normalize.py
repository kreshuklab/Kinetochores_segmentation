import h5py
import numpy as np
from scipy import ndimage
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--datadir', default='', help='Path to data files')

def normalize_edt_target(datadir):
    """
    datadir: path to data files
    """
    for filenum in range(18):
        file = h5py.File(prefix + 'EGFP_' + str(filenum) + '_new.h5', 'r')
        label = file['label']
        raw = file['raw']

        inv_label = np.logical_not(label)
        dt_label = ndimage.distance_transform_edt(inv_label)

        # stdarr = (dt_label - np.mean(dt_label)) / (np.std(dt_label))

        normarr = (dt_label - np.min(dt_label)) / (np.max(dt_label) - np.min(dt_label))
        norm_data = h5py.File(datadir + 'EGFP_' + str(filenum) + '_norm.h5', 'w')
        norm_data.create_dataset('raw', data=raw)
        norm_data.create_dataset('label', data=normarr)
        norm_data.close()

if __name__ == '__main__':
    args = parser.parse_args()
    datadir = args.datadir

    normalize_edt_target(datadir)
