import napari
import numpy as np
import h5py
from skimage.filters import threshold_otsu
import skimage
from skimage import feature
from skimage.segmentation import watershed



predictions = h5py.File(input_file,'r')['predictions']

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

# Threshold the dilated peaks

##### thresh_temp_max = thresh_temp[0] > global_thresh


extra = np.where(thresh_temp != foreground)

thresh_temp[extra] = 0


watershed_output = watershed(thresh_temp, temp_max, mask=thresh_temp)

with napari.gui_qt():
	viewer = napari.Viewer()
	#viewer.add_image()
	viewer.add_labels(watershed_output)