import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.measure import label
import h5py
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('target_path', default='', help='Path for the target h5 file')
parser.add_argument('peak_path', default='', help='Path to the processed 3d Maxima peaks file')
parser.add_argument('sampling' default=[2, 1, 1], help='List of values for voxel spacing')
parser.add_argument('threshold', default=8.0, help='Distance threshold value for selection and matching')

def binarize_peaks(input):
	"""

	input: Input volume to be binarized (processed 3D Maxima over prediction volume)
	"""

	# Get the actual peak indices
	ids = np.nonzero(input)

	# Take an empty volume
	binary_peak = np.zeros((48,128,128))

	# Fill up the indices for sources
	for i in range(len(ids[0])):
		id1, id2, id3 = ids[0][i], ids[1][i], ids[2][i]
		binary_peak[id1][id2][id3] = 1.0

	return binary_peak


def validate_peaks(target, peaks, sampling, distance_threshold):
	"""

	target: binary mask with single pixel sensors
	peaks: maxima from predictions (assuming now that this is binary)
	sampling: the voxel spacing, e.g (2, 1, 1) for anisotropy factor of 2 in z
	"""

	# compute the distance transform of the target
	# the target needs to be inverted here because 0 is treated as foreground
	distances, indices = distance_transform_edt(np.logical_not(target), sampling=sampling, return_indices=True)

	#print(np.unique(distances))

	# apply label to the target, so that we get a unique id for each sensor in the target
	target_labeled = label(target)
	# get the target ids and their position
	target_ids, target_positions = np.unique(target_labeled, return_index=True)

	peaks_labeled = label(peaks)
	peak_ids, peak_positions = np.unique(peaks_labeled, return_index=True)
	# go to 3d positions
	peak_positions = np.unravel_index(peak_positions, distances.shape)

	n_unmatched_peaks = 0
	sensors_to_peaks = {sensor_id : [] for sensor_id in target_ids}

	peak_distances = distances[peak_positions]

	# Indices being a 4D array: (3, 48, 128, 128), thus we need flatten add per dim
	sensor_positions = []
	for ax in range(3):
		sensor_positions_ax = (ax * np.ones_like(peak_positions[0]),) + peak_positions
		sensor_positions.append(indices[sensor_positions_ax])
	sensor_positions = tuple(sensor_positions)

	sensor_ids = target_labeled[sensor_positions]

	for peak_id, distance, sensor_id in zip(peak_ids[1:], peak_distances[1:], sensor_ids[1:]):
	
		if distance > distance_threshold:
			n_unmatched_peaks += 1
			continue

		sensors_to_peaks[sensor_id].append(peak_id)

	n_matched_peaks = 0
	n_overmatched_peaks = 0
	#n_unmatched_sensors = 0

	for x in sensors_to_peaks.values():
		n = len(x)
		if n == 0:
			n_unmatched_sensors += 1
		elif n == 1:
			n_matched_peaks += 1
		else:
			n_overmatched_peaks += (n - 1)

	print(n_matched_peaks)
	print(n_overmatched_peaks)
	print(n_unmatched_sensors)

	print(sensors_to_peaks[])


# Nonzero provides similar output from the complete volume as unique and unravel above
# t_nonzero = np.nonzero(target)
# pk_nonzero = np.nonzero(binary_peak)


def runner_init(target, peak):
	#target_data = h5py.File('/home/samudre/embldata/EGFP/new_processed/train/EGFP_0_new.h5', 'r')['label']
	target_data = h5py.File(target_path, 'r')['label']
	target = target_data[0]

	peak_data = h5py.File(peak_path, 'r')['channel1']
	binary_peak = binarize_peaks(peak_data)

	return target, binary_peak


if __name__ == '__main__':
	args = parser.parse_args()

	target, peak = runner_init(args.target_path, args.peak_path)

	validate_peaks(target, peak, args.sampling, args.threshold)