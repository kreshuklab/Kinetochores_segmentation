import numpy as np
import h5py

from scipy.spatial import distance

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

    def eval_score(self, gt_labels, local_max, watershed_output):

        wshed_peaks = []
        wshed = np.where(watershed_output[0] != 0)

        for inst_id, instance in enumerate(wshed[0]):
            wshed_peaks.append([wshed[0][inst_id], wshed[1][inst_id], wshed[2][inst_id]])


        intersection_peaks = []
        for each in local_max:
            z, y, x = each[0], each[1], each[2]
            for ws in wshed_peaks:
                wsz, wsy, wsx = ws[0], ws[1], ws[2]
                if z == wsz and y == wsy and x == wsx:
                    intersection_peaks.append(ws)

        gt_foreground = np.where(gt_labels[0] == 1.0)
        gt_coords = []

        for idx, val in enumerate(gt_foreground[0]):
            gt_coords.append([gt_foreground[0][idx], gt_foreground[1][idx], gt_foreground[2][idx]])

        eval_dict_fn = self.matching(gt_coords, intersection_peaks)

        fn_count, tp_count = self.get_count(eval_dict_fn)

        print('Total GT peaks: 79')
        print('Prediction peaks after thresholding: ' + str(len(intersection_peaks)))
        print('Matched GT and peaks (TP): ' +  str(tp_count))

        print('No prediction peak for GT (FN): ' + str(fn_count))

        eval_dict_fp = self.matching(intersection_peaks, gt_coords)

        fp_count, tp_count = self.get_count(eval_dict_fp)

        print('No GT for a peak (FP): ' + str(fp_count))

    @staticmethod
    def matching(peaks_list_a, peaks_list_b):

        eval_dict = {}

        for pid_a, peak_a in enumerate(peaks_list_a):
            z, y, x = peak_a[0], peak_a[1], peak_a[2]

            eval_dict[(z, y, x)] = {}

            for pid_b, peak_b in peaks_list_b:
                pz, py, px = peak_b[0], peak_b[1], peak_b[2]

                std_euc = distance.seuclidean([z, y, x], [pz, py, px], [3.0, 1.3, 1.3])

                if std_euc <= 6.0:
                    eval_dict[z, y, x][(pz, py, px)] = std_euc

        return eval_dict


    @staticmethod
    def get_count(eval_dict):

        count = 0
        tp_count = 0

        for key, val in eval_dict.items():
            if not val:
                count += 1
            else:
                tp_count += 1

        return count, tp_count
