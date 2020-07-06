"""
Standardization module
"""

import copy
import math

class standardize(object):
    def __init__(self):
        pass

    def standardize_ds(ds_input, val_ds_input):
        """
        ds_input: input dataset (Tensor)
        val_ds_input: validation input dataset (Tensor)

        return standardized input, val input
        """

        for count_ip, subvol in enumerate(ds_input):
            subvol_input = copy.deepcopy(ds_input[count_ip])
            mean_input, var_input = ds_input[count_ip].mean(), ds_input[count_ip].var()

            subvol_input -= mean_input
            subvol_input = abs(subvol_input)

            subvol_input /= math.sqrt(var_input)
            ds_input[count_ip] = subvol_input

        for count_val_ip, subvol in enumerate(val_ds_input):
            val_subvol_input = copy.deepcopy(val_ds_input[count_val_ip])
            val_mean_input, val_var_input = val_ds_input[count_val_ip].mean(), val_ds_input[count_val_ip].var()

            val_subvol_input -= val_mean_input
            val_subvol_input = abs(val_subvol_input)

            val_subvol_input /= math.sqrt(val_var_input)
            val_ds_input[count_val_ip] = val_subvol_input

        return ds_input, val_ds_input
