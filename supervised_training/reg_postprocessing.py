import numpy as np
import h5py

import skimage
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed


class PostProcessing:

    def __init__(self):
        pass

    @staticmethod
    def load_vols(input, target):
        gt_labels = h5py.File(target, 'r')['label']
        predictions = h5py.File(input, 'r')['predictions']

        return gt_labels, predictions

    def prediction_processing(self, input, target):

        predictions = self.load_vols(input, target)
        predictions = predictions[0]

        # invert the minima and maxima
        invert_preds = predictions * -1

        # shift the minima: range between (0,1]
        invert_preds = invert_preds + (np.min(invert_preds) * -1) + (np.min(invert_preds) * -1 / 100)

        # get the local maxima
        local_max = skimage.feature.peak_local_max(invert_preds, min_distance=2)
        invert_preds = np.expand_dims(invert_preds, axis=0)

        # get the threshold and discard lower vals
        global_thresh = threshold_otsu(invert_preds[0])
        foreground = invert_preds[0] > global_thresh

        # assign instances for local max
        temp_max = np.zeros((1, 48, 128, 128))
        for i, each in enumerate(local_max):
            temp_max[0][each[0], each[1], each[2]] = i+1

        # binary instances for foreground minima removal
        binary_temp_max = np.zeros((1, 48, 128, 128))

        for i, each in enumerate(local_max):
            binary_temp_max[0][each[0], each[1], each[2]] = 1

        foreground = np.expand_dims(foreground.astype(np.uint16), axis=0)

        # remove the spurious peaks
        extra = np.where(foreground != binary_temp_max)
        binary_temp_max[extra] = 0

        watershed_output = watershed(binary_temp_max, temp_max, mask=binary_temp_max).astype(np.uint16)

        return watershed_output
