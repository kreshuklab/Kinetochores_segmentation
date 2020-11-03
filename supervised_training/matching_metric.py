import skimage
from skimage import feature
import numpy as np
import h5py
from scipy.spatial import distance
import itertools
from scipy import ndimage
from skimage.filters import threshold_otsu
import cv2
from sklearn.cluster import KMeans
from skimage.morphology import watershed
import copy


class MatchingMetric:

    def create_instances(input_gt, predictions_file):
    	# load ground truth
    	gt_labels = h5py.File(input_gt,'r')['label']

    	# load predictions
    	predictions = h5py.File(predictions_file,'r')['predictions']

    	# Take otsu threshold
    	global_thresh = threshold_otsu(predictions[0])

    	foreground = predictions > global_thresh

    	# Get the local max
    	local_max = skimage.feature.peak_local_max(predictions[0], min_distance=2)

    	# Prepare the vol with peaks
    	temp_max = np.zeros((1,48,128,128))

    	for i, each in enumerate(local_max):
    		temp_max[0][each[0], each[1], each[2]] = i+1


    	# Dilate the peaks
    	inv_temp_max = np.logical_not(temp_max)
    	dist_tr = ndimage.distance_transform_edt(inv_temp_max)

    	thresh_tr = dist_tr > 2

    	thresh_temp = np.logical_not(thresh_tr).astype(np.float64)

    	extra = np.where(thresh_temp != foreground)

    	thresh_temp[extra] = 0


    	#################
    	# Take intersection of local max and the watershed output peaks 
    	###################

    	watershed_output = watershed(thresh_temp, temp_max, mask=thresh_temp).astype(np.uint16)

        return watershed_output


# print(len(np.where(watershed_output[0]!=0)[0]))

    def matching():
        wshed_peaks = []
        wshed = np.where(watershed_output[0]!=0)

        for wid, wval in enumerate(wshed[0]):
        	wshed_peaks.append([wshed[0][wid], wshed[1][wid], wshed[2][wid]])


        intersection_peaks = []
        for each in local_max:
            z,y,x = each[0], each[1], each[2]
            for ws in wshed_peaks:
                wsz, wsy, wsx = ws[0], ws[1], ws[2]
                if z == wsz and y == wsy and x == wsx:
                    intersection_peaks.append(ws)

        # print(len(intersection_peaks))

        gt_foreground = np.where(gt_labels[0]==1.0)
        gt_coords = []

        for idx, val in enumerate(gt_foreground[0]):
            gt_coords.append([gt_foreground[0][idx], gt_foreground[1][idx], gt_foreground[2][idx]])

        eval_dict = {}

        for gtc in gt_coords:
        	gt_z, gt_y, gt_x = gtc[0], gtc[1], gtc[2]
        	eval_dict[(gt_z, gt_y, gt_x)] = {}

        	for a, peak in enumerate(intersection_peaks):
        		z, y, x = peak[0], peak[1], peak[2]

        		std_euc = distance.seuclidean([gt_z, gt_y, gt_x], [z,y,x], [3.0,1.3,1.3])

        		if std_euc <= 6.0:
        			instance_label = watershed_output[0][wshed_peaks[a][0]][wshed_peaks[a][1]][wshed_peaks[a][2]]
        			#eval_dict[gt_z, gt_y, gt_x][(z,y,x)] = {instance_label, std_euc}
        			eval_dict[gt_z, gt_y, gt_x][(z,y,x)] = std_euc

        tp_count = 0
        fn_count = 0

        for k1, v1 in eval_dict.items():
        	if v1 != {}:
        		tp_count += 1
        		v1sorted = {k:v for k,v in sorted(v1.items(), key=lambda item: item[1])}
        		#print(k1, v1sorted)
        	else:
        		fn_count += 1

        print(len(intersection_peaks))
        print(tp_count)

        print(fn_count)

        eval_dict = {}

        fp_count = 0
        tp_count=0

        for a, peak in enumerate(intersection_peaks):
        	z, y, x = peak[0], peak[1], peak[2]
        	eval_dict[(z, y, x)] = {}

        	for gtc in gt_coords:
        		gt_z, gt_y, gt_x = gtc[0], gtc[1], gtc[2]

        		std_euc = distance.seuclidean([z,y,x], [gt_z, gt_y, gt_x], [3.0,1.3,1.3])

        		if std_euc <= 6.0:
        			#instance_label = watershed_output[0][wshed_peaks[a][0]][wshed_peaks[a][1]][wshed_peaks[a][2]]
        			#eval_dict[gt_z, gt_y, gt_x][(z,y,x)] = {instance_label, std_euc}
        			eval_dict[z, y, x][(gt_z,gt_y,gt_x)] = std_euc

        for k1, v1 in eval_dict.items():
        	if v1 != {}:
        		tp_count += 1
        		v1sorted = {k:v for k,v in sorted(v1.items(), key=lambda item: item[1])}
        	else:
        		fp_count += 1

        print(tp_count)
        print(fp_count)

################## gt_coords and wshed_peaks are list of coord ists  ##############
