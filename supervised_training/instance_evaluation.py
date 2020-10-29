import itertools
import numpy as np
import h5py

from scipy import ndimage

from skimage import feature
from skimage.filters import threshold_otsu

import cc3d

class ThresholdMatching:
    """
    ThresholdMatching class
    """

    def __init__(self):
        pass

    def thresholding(self, predictions):
        """
        :predictions: Prediction output from the network

        """

        # global otsu threshold on the predictions
        global_thresh = threshold_otsu(predictions)

        # define the foreground based on threshold
        foreground = predictions > global_thresh

        # get the local peaks from the predictions
        local_max = skimage.feature.peak_local_max(predictions[0], min_distance=5)

        # prepare the vol with peaks
        local_peaks_vol = np.zeros((1, 48, 128, 128))

        for coordinate in local_max:
            local_peaks_vol[0][coordinate[0], coordinate[1], coordinate[2]] = 1.0

        # dilate the peaks
        inv_local_peaks_vol = np.logical_not(local_peaks_vol)

        # get distance transform
        local_peaks_edt = ndimage.distance_transform_edt(inv_local_peaks_vol)

        # threshold the edt and invert back: fg as 1, bg as 0
        spherical_labels = local_peaks_edt > 3
        spherical_labels = np.logical_not(spherical_labels).astype(np.float64)

        # get the outliers based on threshold and set zero
        outliers = np.where(spherical_labels != foreground)
        spherical_labels[outliers] = 0

        return spherical_labels


    def get_instances(self, spherical_labels):
        """
        :spherical_labels: output predictions to be labeled individually 

        """

        spherical_labels = self.thresholding(predictions)
        # connectivity for connected components
        connectivity = 6

        spherical_labels = spherical_labels.astype(np.uint16)

        instances = cc3d.connected_components(spherical_labels[0], connectivity=connectivity)

        instances = instances.astype(float64)
        instances = np.expand_dims(instances, axis=0)

        instances_vol = h5py.File('peak_instances.h5', 'w')
        instances_vol.create_dataset('peaks', data=instances)
        instances_vol.close()
