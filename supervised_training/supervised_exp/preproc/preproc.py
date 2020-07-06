"""
Preprocessing module
"""

import glob
import torch
import h5py

#train_dir = glob.glob('/content/drive/My Drive/EGFP/train/*')
#valid_dir = glob.glob('/content/drive/My Drive/EGFP/val/*')

class preproc(object):
    def __init__(self):
        pass

    def getdata(train_dir_path, valid_dir_path):
        """
        train_dir_path: path to train ds dir (str)
        valid_dir_path: path to valid ds dir (str)

        return dataset dicts
        """

        train_dir = glob.glob(train_dir_path + '*')
        valid_dir = glob.glob(valid_dir_path + '*')

        data_input = {}
        data_target = {}
        val_data_input = {}
        val_data_target = {}

        # Read dataset files
        for each, num in zip(train_dir, range(len(train_dir))):
            in_file = h5py.File(each, 'r')
            data_input[num] = in_file['raw']
            data_target[num] = in_file['label']

        # Similar process for validation data
        for each, num in zip(valid_dir, range(len(valid_dir))):
            in_file = h5py.File(each, 'r')
            val_data_input[num] = in_file['raw']
            val_data_target[num] = in_file['label']

        return data_input, data_target, val_data_input, val_data_target

    def get_tensors(data_input, data_target, val_data_input, val_data_target):
        """
        data_input: input dataset (dict)
        data_target: target dataset (dict)
        val_data_input: validation input dataset (dict)
        val_data_target: validation target dataset (dict)

        return tensors
        """

        ds_input = []
        ds_target = []
        val_ds_input = []
        val_ds_target = []

        for k, v in data_input.items():
            ds_input.append(v)

        for k, v in data_target.items():
            ds_target.append(v)

        for k, v in val_data_input.items():
            val_ds_input.append(v)

        for k, v in val_data_target.items():
            val_ds_target.append(v)

        ds_input = torch.Tensor(ds_input)
        ds_target = torch.Tensor(ds_target)
        val_ds_input = torch.Tensor(val_ds_input)
        val_ds_target = torch.Tensor(val_ds_target)

        return ds_input, ds_target, val_ds_input, val_ds_target
