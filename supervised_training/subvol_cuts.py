"""Transform annotation as per subvol"""

import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('annotation_pickle_path', default='', help='Path to annotation pickle file')
parser.add_argument('new_filename', default='', help='New filename')

def subvol_annotations(old_fname, new_fname):
    """Get the vol and transform
    as per the subvol size
    using 12-60 out of 61 for Z
    (12 start for Z), 436-564
    for Y, X (436 as start for Y, X)."""

    with open(old_fname, 'rb') as hn:
        data = pickle.load(hn)

    newdata = {}
    for k, val in data.items():
        val_list = []
        val_list.append(val[0] - 436)
        val_list.append(val[1] - 436)
        val_list.append(val[2] - 12)
        newdata[k] = val_list

    with open(new_fname, 'wb') as handle:
        pickle.dump(newdata, handle)

    return new_fname

def ds_creator(pickle_file):
    """Get the h5 datatset
    as the format mentioned
    in the pytorch-3dunet"""

    with open(pickle_file, 'rb') as new_pkl:
        labels = pickle.load(new_pkl)

    assert isinstance(labels, dict)

    for k, v in labels.items():
        v[0], v[1], v[2] = int(v[0]), int(v[1]), int(v[2])
        labels[v[2], v[1], v[0]] = 1.0

    label_h5_name = 'raw' + pickle_file.split('.')[0][-2:] + '.h5'
    label_h5 = h5py.File(label_h5_name, 'w')
    label_h5.create_dataset('raw', data=labels)

if __name__ == '__main__':

    args = parser.parse_args()
    pickle_file = subvol_annotations(args.annotation_pickle_path, args.new_filename)

    ds_creator(pickle_file)
