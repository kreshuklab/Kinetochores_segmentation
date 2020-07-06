"""Data preprocessing module """

import copy
import math
import torch
import numpy as np

def data_reshape(data, val_data):
    """reshape data in 
    C,Z,Y,X format"""

    ds_target = []
    val_ds_target = []

    for k, v in data.items():
        ds_target.append(np.transpose(v, (3, 0, 1, 2)))

    for k, v in val_data.items():
        val_ds_target.append(np.transpose(v, (3, 0, 1, 2)))

    ds_target = torch.Tensor(ds_target)
    val_ds_target = torch.Tensor(val_ds_target)

    return ds_target, val_ds_target


# Standardize the data

def data_standardize(ds_target, val_ds_target):
    """Standardize the signal 
        values in the volume"""

    for num in enumerate(ds_target):
        subvol_target = copy.deepcopy(ds_target[num])
        mean_target, var_target = ds_target[num].mean(), ds_target[num].var()

        subvol_target -= mean_target
        subvol_target = abs(subvol_target)

        subvol_target /= math.sqrt(var_target)
        ds_target[num] = subvol_target

    for v_num in enumerate(val_ds_target):
        val_subvol_target = copy.deepcopy(val_ds_target[v_num])
        val_mean_target, val_var_target = val_ds_target[v_num].mean(), val_ds_target[v_num].var()

        val_subvol_target -= val_mean_target
        val_subvol_target = abs(val_subvol_target)

        val_subvol_target /= math.sqrt(val_var_target)
        val_ds_target[v_num] = val_subvol_target

    ds_input = copy.deepcopy(ds_target)
    val_ds_input = copy.deepcopy(val_ds_target)

    # print(ds_input.shape)

    # Blank out the middle slice
    for num in enumerate(ds_input):
        ds_input[num][0][15:16, :, :] = 0.0

    for v_num in enumerate(val_ds_input):
        val_ds_input[v_num][0][15:16, :, :] = 0.0

    return ds_input, val_ds_input, ds_target, val_ds_target
