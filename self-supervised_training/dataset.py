"""Dataset module"""

import os
import h5py

def load_dataset(data_dir_path):
    """Load the dataset (HDF5 file) 
    from the path """

    data_dir = os.listdir(data_dir_path)

    train_dir = data_dir[:8]
    valid_dir = data_dir[8:12]

    data = {}
    val_data = {}

    for each, num in enumerate(train_dir):
        in_file = h5py.File(each, 'r')
        data[num] = in_file['exported_data'][16:48, 436:564, 436:564, :].astype('float32')

    # Similar process for validation data

    for each, num in enumerate(valid_dir):
        in_file = h5py.File(each, 'r')
        val_data[num] = in_file['exported_data'][16:48, 436:564, 436:564, :].astype('float32')

    return data, val_data
