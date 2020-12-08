import importlib
import os
import time

import hdbscan
import numpy as np
import torch
import skimage
from skimage import feature
from skimage import measure
from skimage.filters import threshold_otsu
from skimage.metrics import adapted_rand_error, peak_signal_noise_ratio, mean_squared_error
from skimage.metrics import normalized_root_mse
from skimage.segmentation import watershed

from sklearn.cluster import MeanShift
from scipy.spatial import distance
from scipy import ndimage

from pytorch3dunet.unet3d.losses import compute_per_channel_dice
from pytorch3dunet.unet3d.seg_metrics import AveragePrecision, Accuracy
from pytorch3dunet.unet3d.utils import get_logger, expand_as_one_hot, plot_segm, convert_to_numpy

logger = get_logger('EvalMetric')


class DiceCoefficient:
	"""Computes Dice Coefficient.
	Generalized to multiple channels by computing per-channel Dice Score
	(as described in https://arxiv.org/pdf/1707.03237.pdf) and theTn simply taking the average.
	Input is expected to be probabilities instead of logits.
	This metric is mostly useful when channels contain the same semantic class (e.g. affinities computed with different offsets).
	DO NOT USE this metric when training with DiceLoss, otherwise the results will be biased towards the loss.
	"""

	def __init__(self, epsilon=1e-6, **kwargs):
		self.epsilon = epsilon

	def __call__(self, input, target):
		# Average across channels in order to get the final score
		return torch.mean(compute_per_channel_dice(input, target, epsilon=self.epsilon))

##############################################################
########################## REVIEW ############################
##############################################################


class MultiHead_metric:

    def __init__(self, skip_channels=(), ignore_index=None, **kwargs):
        self.ignore_index = ignore_index
        self.skip_channels = skip_channels

    def __call__(self, input1, input2, target1, target2):
        MIou = MeanIoU()
        metric_seg = MIou(input1, target1)

        rmse = RMSE()
        metric_reg = rmse(input2, target2)

        return metric_seg + metric_reg

class MeanIoU:
	"""
	Computes IoU for each class separately and then averages over all classes.
	"""

	def __init__(self, skip_channels=(), ignore_index=None, **kwargs):
		"""
		:param skip_channels: list/tuple of channels to be ignored from the IoU computation
		:param ignore_index: id of the label to be ignored from IoU computation
		"""
		self.ignore_index = ignore_index
		self.skip_channels = skip_channels

	def __call__(self, input, target):
		"""
		:param input: 5D probability maps torch float tensor (NxCxDxHxW)
		:param target: 4D or 5D ground truth torch tensor. 4D (NxDxHxW) tensor will be expanded to 5D as one-hot
		:return: intersection over union averaged over all channels
		"""
		assert input.dim() == 5

		n_classes = input.size()[1]

		if target.dim() == 4:
			target = expand_as_one_hot(target, C=n_classes, ignore_index=self.ignore_index)

		assert input.size() == target.size()

		per_batch_iou = []
		for _input, _target in zip(input, target):
			binary_prediction = self._binarize_predictions(_input, n_classes)

			if self.ignore_index is not None:
				# zero out ignore_index
				mask = _target == self.ignore_index
				binary_prediction[mask] = 0
				_target[mask] = 0

			# convert to uint8 just in case
			binary_prediction = binary_prediction.byte()
			_target = _target.byte()

			per_channel_iou = []
			for c in range(n_classes):
				if c in self.skip_channels:
					continue

				per_channel_iou.append(self._jaccard_index(binary_prediction[c], _target[c]))

			assert per_channel_iou, "All channels were ignored from the computation"
			mean_iou = torch.mean(torch.tensor(per_channel_iou))
			per_batch_iou.append(mean_iou)

		return torch.mean(torch.tensor(per_batch_iou))

	def _binarize_predictions(self, input, n_classes):
		"""
		Puts 1 for the class/channel with the highest probability and 0 in other channels. Returns byte tensor of the
		same size as the input tensor.
		"""
		if n_classes == 1:
			# for single channel input just threshold the probability map
			result = input > 0.5
			return result.long()

		_, max_index = torch.max(input, dim=0, keepdim=True)
		return torch.zeros_like(input, dtype=torch.uint8).scatter_(0, max_index, 1)

	def _jaccard_index(self, prediction, target):
		"""
		Computes IoU for a given target and prediction tensors
		"""
		return torch.sum(prediction & target).float() / torch.clamp(torch.sum(prediction | target).float(), min=1e-8)


##############################################################
##############################################################







class PeakMatching:
	"""
	Metric for local peak - maxima/ minima in a specified radius region
	"""

	def __init__(self, **kwargs):
		pass

	def __call__(self, input, target):
		"""
		:input: The predictions from the network (BS*C*D*H*W)
		:target: The ground truth passed to the network (BS*C*D*H*W)
		:return: Number of matched peaks
		"""

		# look for local max within fixed range between input and target

		# Get input and predictions in required format
		input, target = convert_to_numpy(input, target)
		
		# pass the original gt coordinates here
		#original_gt = #

		# Take the peaks from the predictions
		local_max = skimage.feature.peak_local_max(input[0][0], min_distance=4)

		# Take the foreground pixels from ground truth
		foreground = np.where(target[0][0]==1.0)
		foreground_coords = []

		for idx, val in enumerate(foreground[0]):
			foreground_coords.append([foreground[0][idx], foreground[1][idx], foreground[2][idx]])

		eval_dict = {}

		# Matching procedure
		for coord in local_max:
			z, y, x = coord[0], coord[1], coord[2]
			eval_dict[(z,y,x)] = {}

			for coord in foreground_coords:
				gt_z, gt_y, gt_x = coord[0], coord[1], coord[2]

				# Taking anisotropy into account
				std_euc = distance.seuclidean([z,y,x], [gt_z, gt_y, gt_x], [3.0,1.3,1.3])

				# Keeping a threshold for matching
				if std_euc <=8.0:
					eval_dict[(z,y,x)][gt_z, gt_y, gt_x] = std_euc

		n_count = 0
		for k, v in eval_dict.items():
			if v != {}:
				#vsorted = {k2: v2 for k2, v2 in sorted(v.items(), key=lambda item: item[1])}

				n_count += 1
				#print(k, dict(itertools.islice(vsorted.items(), 1)))
		
		print(n_count)
		return torch.tensor(n_count)


class Thresh_IoU:
	"""
	Computes IoU for each class separately and then averages over all classes.
	"""

	def __init__(self, skip_channels=(), ignore_index=None, **kwargs):
		"""
		:param skip_channels: list/tuple of channels to be ignored from the IoU computation
		:param ignore_index: id of the label to be ignored from IoU computation
		"""
		self.ignore_index = ignore_index
		self.skip_channels = skip_channels

	def __call__(self, input, target):
		"""
		:param input: 5D probability maps torch float tensor (NxCxDxHxW)
		:param target: 4D or 5D ground truth torch tensor. 4D (NxDxHxW) tensor will be expanded to 5D as one-hot
		:return: intersection over union averaged over all channels
		"""
		assert input.dim() == 5

		predictions, target = convert_to_numpy(input, target)
		predictions = predictions[0]

		# global otsu threshold on the predictions
		global_thresh = threshold_otsu(predictions)

		low_intensity_region = np.where(predictions < global_thresh)

		predictions = np.array(predictions)
		predictions[low_intensity_region] = 0
		predictions = np.expand_dims(predictions, axis=0)

		predictions = torch.tensor(predictions)

		target = torch.tensor(target)

		n_classes = input.size()[1]

		if target.dim() == 4:
			target = expand_as_one_hot(target, C=n_classes, ignore_index=self.ignore_index)

		assert predictions.size() == target.size()

		per_batch_iou = []
		for _input, _target in zip(predictions, target):
			binary_prediction = self._binarize_predictions(_input, n_classes)

			if self.ignore_index is not None:
				# zero out ignore_index
				mask = _target == self.ignore_index
				binary_prediction[mask] = 0
				_target[mask] = 0

			# convert to uint8 just in case
			binary_prediction = binary_prediction.byte()
			_target = _target.byte()

			per_channel_iou = []
			for c in range(n_classes):
				if c in self.skip_channels:
					continue

				per_channel_iou.append(self._jaccard_index(binary_prediction[c], _target[c]))

			assert per_channel_iou, "All channels were ignored from the computation"
			mean_iou = torch.mean(torch.tensor(per_channel_iou))
			per_batch_iou.append(mean_iou)

		return torch.mean(torch.tensor(per_batch_iou))

	def _binarize_predictions(self, input, n_classes):
		"""
		Puts 1 for the class/channel with the highest probability and 0 in other channels. Returns byte tensor of the
		same size as the input tensor.
		"""
		if n_classes == 1:
			# for single channel input just threshold the probability map
			result = input > 0.5
			return result.long()

		_, max_index = torch.max(input, dim=0, keepdim=True)
		return torch.zeros_like(input, dtype=torch.uint8).scatter_(0, max_index, 1)

	def _jaccard_index(self, prediction, target):
		"""
		Computes IoU for a given target and prediction tensors
		"""
		return torch.sum(prediction & target).float() / torch.clamp(torch.sum(prediction | target).float(), min=1e-8)


class Peaks_IoU:
	"""
	Computes IoU for each class separately and then averages over all classes.
	"""

	def __init__(self, skip_channels=(), ignore_index=None, **kwargs):
		"""
		:param skip_channels: list/tuple of channels to be ignored from the IoU computation
		:param ignore_index: id of the label to be ignored from IoU computation
		"""
		self.ignore_index = ignore_index
		self.skip_channels = skip_channels

	def __call__(self, input, target):
		"""
		:param input: 5D probability maps torch float tensor (NxCxDxHxW)
		:param target: 4D or 5D ground truth torch tensor. 4D (NxDxHxW) tensor will be expanded to 5D as one-hot
		:return: intersection over union averaged over all channels
		"""
		assert input.dim() == 5

		predictions, target = convert_to_numpy(input, target)
		predictions = predictions[0]

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
		spherical_peaks = local_peaks_edt > 3
		spherical_peaks = np.logical_not(spherical_peaks).astype(np.float64)

		# get the outliers based on threshold and set zero
		outliers = np.where(spherical_peaks != foreground)
		spherical_peaks[outliers] = 0

		spherical_peaks = np.expand_dims(spherical_peaks, axis=0)
		# print(spherical_peaks.shape)
		# print(np.min(spherical_peaks))
		# print(np.max(spherical_peaks))
		# print(len(np.where(spherical_peaks==1.0)[0]))

		# spherical_peaks = torch.tensor(spherical_peaks)

		# spherical_peaks.to('cuda')

		spherical_peaks = torch.tensor(spherical_peaks)
		target = torch.tensor(target)

		n_classes = input.size()[1]

		if target.dim() == 4:
			target = expand_as_one_hot(target, C=n_classes, ignore_index=self.ignore_index)

		assert spherical_peaks.size() == target.size()

		per_batch_iou = []
		for _input, _target in zip(spherical_peaks, target):
			binary_prediction = self._binarize_predictions(_input, n_classes)

			if self.ignore_index is not None:
				# zero out ignore_index
				mask = _target == self.ignore_index
				binary_prediction[mask] = 0
				_target[mask] = 0

			# convert to uint8 just in case
			binary_prediction = binary_prediction.byte()
			_target = _target.byte()

			per_channel_iou = []
			for c in range(n_classes):
				if c in self.skip_channels:
					continue

				per_channel_iou.append(self._jaccard_index(binary_prediction[c], _target[c]))

			assert per_channel_iou, "All channels were ignored from the computation"
			mean_iou = torch.mean(torch.tensor(per_channel_iou))
			per_batch_iou.append(mean_iou)

		return torch.mean(torch.tensor(per_batch_iou))


	def _binarize_predictions(self, input, n_classes):
		"""
		Puts 1 for the class/channel with the highest probability and 0 in other channels. Returns byte tensor of the
		same size as the input tensor.
		"""
		if n_classes == 1:
			# for single channel input just threshold the probability map
			result = input > 0.5
			return result.long()

		_, max_index = torch.max(input, dim=0, keepdim=True)
		return torch.zeros_like(input, dtype=torch.uint8).scatter_(0, max_index, 1)

	def _jaccard_index(self, prediction, target):
		"""
		Computes IoU for a given target and prediction tensors
		"""
		return torch.sum(prediction & target).float() / torch.clamp(torch.sum(prediction | target).float(), min=1e-8)


class instance_count:

	def __init__(self, **kwargs):
		pass

	def precision(self, tp_count, fp_count):
		return tp_count / (tp_count + fp_count)

	def recall(self, tp_count, fn_count):
		return tp_count / (tp_count + fn_count)

	def __call__(self, input, target):
		predictions, target = convert_to_numpy(input, target)

		predictions = predictions[0]
		target = target[0]

		global_thresh = threshold_otsu(predictions[0])

		foreground = predictions > global_thresh

		local_max = skimage.feature.peak_local_max(predictions[0], min_distance=3)

		temp_max = np.zeros((1,48,128,128))

		for i, each in enumerate(local_max):
			temp_max[0][each[0], each[1], each[2]] = i+1

		inv_temp_max = np.logical_not(temp_max)
		dist_tr = ndimage.distance_transform_edt(inv_temp_max)

		# Thresh val.
		thresh_tr = dist_tr > 2

		thresh_temp = np.logical_not(thresh_tr).astype(np.float64)

		extra = np.where(thresh_temp != foreground)

		thresh_temp[extra] = 0

		watershed_output = watershed(thresh_temp, temp_max, mask=thresh_temp).astype(np.uint16)

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

		gt_foreground = np.where(target[0]==1.0)
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
					#instance_label = watershed_output[0][wshed_peaks[a][0]][wshed_peaks[a][1]][wshed_peaks[a][2]]
					#eval_dict[gt_z, gt_y, gt_x][(z,y,x)] = {instance_label, std_euc}
					eval_dict[gt_z, gt_y, gt_x][(z,y,x)] = std_euc

		tp_count = 0
		fn_count = 0

		for k1, v1 in eval_dict.items():
			if v1 != {}:
				tp_count += 1
				#v1sorted = {k:v for k,v in sorted(v1.items(), key=lambda item: item[1])}
				#print(k1, v1sorted)
			else:
				fn_count += 1

		print('Total GT peaks: 79')
		print('Prediction peaks after thresholding: ' + str(len(intersection_peaks)))
		print('Matched GT and peaks (TP): ' +  str(tp_count))

		# fn count -> 

		print('No prediction peak for GT (FN): ' + str(fn_count))
		# for k, v in eval_dict.items():
		#   print(k, v)

		eval_dict = {}

		fp_count = 0
		tp_count = 0

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
				#v1sorted = {k:v for k,v in sorted(v1.items(), key=lambda item: item[1])}
				#print(k1, v1sorted)
			else:
				fp_count += 1
				#print(k1)

		#print('' + tp_count)
		print('No GT for a peak (FP): ' + str(fp_count))

		precision_val = self.precision(tp_count, fp_count)
		recall_val = self.recall(tp_count, fn_count)

		F1_score = 2 * precision_val * recall_val / (precision_val + recall_val)

		return torch.tensor(F1_score)

class AdaptedRandError:
	"""
	A functor which computes an Adapted Rand error as defined by the SNEMI3D contest
	(http://brainiac2.mit.edu/SNEMI3D/evaluation).

	This is a generic implementation which takes the input, converts it to the segmentation image (see `input_to_segm()`)
	and then computes the ARand between the segmentation and the ground truth target. Depending on one's use case
	it's enough to extend this class and implement the `input_to_segm` method.

	Args:
		use_last_target (bool): use only the last channel from the target to compute the ARand
		save_plots (bool): save predicted segmentation (result from `input_to_segm`) together with GT segmentation as a PNG
		plots_dir (string): directory where the plots are to be saved
	"""

	def __init__(self, use_last_target=False, save_plots=False, plots_dir='.', **kwargs):
		self.use_last_target = use_last_target
		self.save_plots = save_plots
		self.plots_dir = plots_dir
		if not os.path.exists(plots_dir) and save_plots:
			os.makedirs(plots_dir)

	def __call__(self, input, target):
		"""
		Compute ARand Error for each input, target pair in the batch and return the mean value.

		Args:
			input (torch.tensor): 5D (NCDHW) output from the network
			target (torch.tensor): 4D (NDHW) ground truth segmentation

		Returns:
			average ARand Error across the batch
		"""
		def _arand_err(gt, seg):
			n_seg = len(np.unique(seg))
			if n_seg == 1:
				return 0.
			return adapted_rand_error(gt, seg)[0]

		# converts input and target to numpy arrays
		input, target = convert_to_numpy(input, target)
		if self.use_last_target:
			target = target[:, -1, ...]  # 4D
		else:
			# use 1st target channel
			target = target[:, 0, ...]  # 4D

		# ensure target is of integer type
		target = target.astype(np.int)

		per_batch_arand = []
		for _input, _target in zip(input, target):
			n_clusters = len(np.unique(_target))
			# skip ARand eval if there is only one label in the patch due to the zero-division error in Arand impl
			# xxx/skimage/metrics/_adapted_rand_error.py:70: RuntimeWarning: invalid value encountered in double_scalars
			# precision = sum_p_ij2 / sum_a2
			logger.info(f'Number of ground truth clusters: {n_clusters}')
			if n_clusters == 1:
				logger.info('Skipping ARandError computation: only 1 label present in the ground truth')
				per_batch_arand.append(0.)
				continue

			# convert _input to segmentation CDHW
			segm = self.input_to_segm(_input)
			assert segm.ndim == 4

			if self.save_plots:
				# save predicted and ground truth segmentation
				plot_segm(segm, _target, self.plots_dir)

			# compute per channel arand and return the minimum value
			per_channel_arand = [_arand_err(_target, channel_segm) for channel_segm in segm]
			logger.info(f'Min ARand for channel: {np.argmin(per_channel_arand)}')
			per_batch_arand.append(np.min(per_channel_arand))

		# return mean arand error
		mean_arand = torch.mean(torch.tensor(per_batch_arand))
		logger.info(f'ARand: {mean_arand.item()}')
		return mean_arand

	def input_to_segm(self, input):
		"""
		Converts input tensor (output from the network) to the segmentation image. E.g. if the input is the boundary
		pmaps then one option would be to threshold it and run connected components in order to return the segmentation.

		:param input: 4D tensor (CDHW)
		:return: segmentation volume either 4D (segmentation per channel)
		"""
		# by deafult assume that input is a segmentation volume itself
		return input


class BoundaryAdaptedRandError(AdaptedRandError):
	"""
	Compute ARand between the input boundary map and target segmentation.
	Boundary map is thresholded, and connected components is run to get the predicted segmentation
	"""

	def __init__(self, thresholds=None, use_last_target=True, input_channel=None, invert_pmaps=True,
				 save_plots=False, plots_dir='.', **kwargs):
		super().__init__(use_last_target=use_last_target, save_plots=save_plots, plots_dir=plots_dir, **kwargs)
		if thresholds is None:
			thresholds = [0.3, 0.4, 0.5, 0.6]
		assert isinstance(thresholds, list)
		self.thresholds = thresholds
		self.input_channel = input_channel
		self.invert_pmaps = invert_pmaps

	def input_to_segm(self, input):
		if self.input_channel is not None:
			input = np.expand_dims(input[self.input_channel], axis=0)

		segs = []
		for predictions in input:
			for th in self.thresholds:
				# threshold probability maps
				predictions = predictions > th

				if self.invert_pmaps:
					# for connected component analysis we need to treat boundary signal as background
					# assign 0-label to boundary mask
					predictions = np.logical_not(predictions)

				predictions = predictions.astype(np.uint8)
				# run connected components on the predicted mask; consider only 1-connectivity
				seg = measure.label(predictions, background=0, connectivity=1)
				segs.append(seg)

		return np.stack(segs)


class GenericAdaptedRandError(AdaptedRandError):
	def __init__(self, input_channels, thresholds=None, use_last_target=True, invert_channels=None,
				 save_plots=False, plots_dir='.', **kwargs):

		super().__init__(use_last_target=use_last_target, save_plots=save_plots, plots_dir=plots_dir, **kwargs)
		assert isinstance(input_channels, list) or isinstance(input_channels, tuple)
		self.input_channels = input_channels
		if thresholds is None:
			thresholds = [0.3, 0.4, 0.5, 0.6]
		assert isinstance(thresholds, list)
		self.thresholds = thresholds
		if invert_channels is None:
			invert_channels = []
		self.invert_channels = invert_channels

	def input_to_segm(self, input):
		# pick only the channels specified in the input_channels
		results = []
		for i in self.input_channels:
			c = input[i]
			# invert channel if necessary
			if i in self.invert_channels:
				c = 1 - c
			results.append(c)

		input = np.stack(results)

		segs = []
		for predictions in input:
			for th in self.thresholds:
				# run connected components on the predicted mask; consider only 1-connectivity
				seg = measure.label((predictions > th).astype(np.uint8), background=0, connectivity=1)
				segs.append(seg)

		return np.stack(segs)


class EmbeddingsAdaptedRandError(AdaptedRandError):
	def __init__(self, min_cluster_size=100, min_samples=None, metric='euclidean', cluster_selection_method='eom',
				 save_plots=False, plots_dir='.', **kwargs):
		super().__init__(save_plots=save_plots, plots_dir=plots_dir, **kwargs)

		logger.info(f'HDBSCAN params: min_cluster_size: {min_cluster_size}, min_samples: {min_samples}')
		self.clustering = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric=metric,
										  cluster_selection_method=cluster_selection_method)

	def input_to_segm(self, embeddings):
		logger.info("Computing clusters with HDBSCAN...")

		# shape of the output segmentation
		output_shape = embeddings.shape[1:]
		# reshape (C, D, H, W) -> (C, D * H * W) and transpose
		flattened_embeddings = embeddings.reshape(embeddings.shape[0], -1).transpose()

		# perform clustering and reshape in order to get the segmentation volume
		start = time.time()
		segm = self.clustering.fit_predict(flattened_embeddings).reshape(output_shape)
		logger.info(f'Number of clusters found by HDBSCAN: {np.max(segm)}. Duration: {time.time() - start} sec.')

		# assign noise to new cluster (by default hdbscan gives -1 label to outliers)
		noise_label = np.max(segm) + 1
		segm[segm == -1] = noise_label

		return np.expand_dims(segm, axis=0)


# Just for completeness, however sklean MeanShift implementation is just too slow for clustering embeddings
class EmbeddingsMeanShiftAdaptedRandError(AdaptedRandError):
	def __init__(self, bandwidth, save_plots=False, plots_dir='.', **kwargs):
		super().__init__(save_plots=save_plots, plots_dir=plots_dir, **kwargs)
		logger.info(f'MeanShift params: bandwidth: {bandwidth}')
		# use bin_seeding to speedup the mean-shift significantly
		self.clustering = MeanShift(bandwidth=bandwidth, bin_seeding=True)

	def input_to_segm(self, embeddings):
		logger.info("Computing clusters with MeanShift...")

		# shape of the output segmentation
		output_shape = embeddings.shape[1:]
		# reshape (C, D, H, W) -> (C, D * H * W) and transpose
		flattened_embeddings = embeddings.reshape(embeddings.shape[0], -1).transpose()

		# perform clustering and reshape in order to get the segmentation volume
		start = time.time()
		segm = self.clustering.fit_predict(flattened_embeddings).reshape(output_shape)
		logger.info(f'Number of clusters found by MeanShift: {np.max(segm)}. Duration: {time.time() - start} sec.')
		return np.expand_dims(segm, axis=0)


class GenericAveragePrecision:
	def __init__(self, min_instance_size=None, use_last_target=False, metric='ap', **kwargs):
		self.min_instance_size = min_instance_size
		self.use_last_target = use_last_target
		assert metric in ['ap', 'acc']
		if metric == 'ap':
			# use AveragePrecision
			self.metric = AveragePrecision()
		else:
			# use Accuracy at 0.5 IoU
			self.metric = Accuracy(iou_threshold=0.5)

	def __call__(self, input, target):
		assert isinstance(input, torch.Tensor) and isinstance(target, torch.Tensor)
		assert input.dim() == 5
		assert target.dim() == 5

		input, target = convert_to_numpy(input, target)
		if self.use_last_target:
			target = target[:, -1, ...]  # 4D
		else:
			# use 1st target channel
			target = target[:, 0, ...]  # 4D

		batch_aps = []
		# iterate over the batch
		for inp, tar in zip(input, target):
			segs = self.input_to_seg(inp)  # 4D
			# convert target to seg
			tar = self.target_to_seg(tar)
			# filter small instances if necessary
			tar = self._filter_instances(tar)

			# compute average precision per channel
			segs_aps = [self.metric(self._filter_instances(seg), tar) for seg in segs]

			logger.info(f'Max Average Precision for channel: {np.argmax(segs_aps)}')
			# save max AP
			batch_aps.append(np.max(segs_aps))

		return torch.tensor(batch_aps).mean()

	def _filter_instances(self, input):
		"""
		Filters instances smaller than 'min_instance_size' by overriding them with 0-index
		:param input: input instance segmentation
		"""
		if self.min_instance_size is not None:
			labels, counts = np.unique(input, return_counts=True)
			for label, count in zip(labels, counts):
				if count < self.min_instance_size:
					input[input == label] = 0
		return input

	def input_to_seg(self, input):
		raise NotImplementedError

	def target_to_seg(self, target):
		return target


class BlobsAveragePrecision(GenericAveragePrecision):
	"""
	Computes Average Precision given foreground prediction and ground truth instance segmentation.
	"""

	def __init__(self, thresholds=None, metric='ap', min_instance_size=None, input_channel=0, **kwargs):
		super().__init__(min_instance_size=min_instance_size, use_last_target=True, metric=metric)
		if thresholds is None:
			thresholds = [0.4, 0.5, 0.6, 0.7, 0.8]
		assert isinstance(thresholds, list)
		self.thresholds = thresholds
		self.input_channel = input_channel

	def input_to_seg(self, input):
		input = input[self.input_channel]
		segs = []
		for th in self.thresholds:
			# threshold and run connected components
			mask = (input > th).astype(np.uint8)
			seg = measure.label(mask, background=0, connectivity=1)
			segs.append(seg)
		return np.stack(segs)


class BlobsBoundaryAveragePrecision(GenericAveragePrecision):
	"""
	Computes Average Precision given foreground prediction, boundary prediction and ground truth instance segmentation.
	Segmentation mask is computed as (P_mask - P_boundary) > th followed by a connected component
	"""
	def __init__(self, thresholds=None, metric='ap', min_instance_size=None, **kwargs):
		super().__init__(min_instance_size=min_instance_size, use_last_target=True, metric=metric)
		if thresholds is None:
			thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
		assert isinstance(thresholds, list)
		self.thresholds = thresholds

	def input_to_seg(self, input):
		# input = P_mask - P_boundary
		input = input[0] - input[1]
		segs = []
		for th in self.thresholds:
			# threshold and run connected components
			mask = (input > th).astype(np.uint8)
			seg = measure.label(mask, background=0, connectivity=1)
			segs.append(seg)
		return np.stack(segs)


class BoundaryAveragePrecision(GenericAveragePrecision):
	"""
	Computes Average Precision given boundary prediction and ground truth instance segmentation.
	"""

	def __init__(self, thresholds=None, min_instance_size=None, input_channel=0, **kwargs):
		super().__init__(min_instance_size=min_instance_size, use_last_target=True)
		if thresholds is None:
			thresholds = [0.3, 0.4, 0.5, 0.6]
		assert isinstance(thresholds, list)
		self.thresholds = thresholds
		self.input_channel = input_channel

	def input_to_seg(self, input):
		input = input[self.input_channel]
		segs = []
		for th in self.thresholds:
			seg = measure.label(np.logical_not(input > th).astype(np.uint8), background=0, connectivity=1)
			segs.append(seg)
		return np.stack(segs)


class PSNR:
	"""
	Computes Peak Signal to Noise Ratio. Use e.g. as an eval metric for denoising task
	"""

	def __init__(self, **kwargs):
		pass

	def __call__(self, input, target):
		input, target = convert_to_numpy(input, target)
		return peak_signal_noise_ratio(target, input)

class MSE:
    """
    Compute the Mean Squared Error (Regression tasks).
    """

    def __init__(self, **kwargs):
        pass

    def __call__(self, input, target):
        input, target = convert_to_numpy(input, target)
        return mean_squared_error(target, input)

class RMSE:
	"""
	Compute the Root Mean Squared Error (Regression tasks).
	"""

	def __init__(self, **kwargs):
		pass

	def __call__(self, input, target):
		input, target = convert_to_numpy(input, target)
		return normalized_root_mse(target, input)

class WithinAngleThreshold:
	"""
	Returns the percentage of predicted directions which are more than 'angle_threshold' apart from the ground
	truth directions. 'angle_threshold' is expected to be given in degrees not radians.
	"""

	def __init__(self, angle_threshold, **kwargs):
		self.threshold_radians = angle_threshold / 360 * np.pi

	def __call__(self, inputs, targets):
		assert isinstance(inputs, list)
		if len(inputs) == 1:
			targets = [targets]
		assert len(inputs) == len(targets)

		within_count = 0
		total_count = 0
		for input, target in zip(inputs, targets):
			# normalize and multiply by the stability_coeff in order to prevent NaN results from torch.acos
			stability_coeff = 0.999999
			input = input / torch.norm(input, p=2, dim=1).detach().clamp(min=1e-8) * stability_coeff
			target = target / torch.norm(target, p=2, dim=1).detach().clamp(min=1e-8) * stability_coeff
			# compute cosine map
			cosines = (input * target).sum(dim=1)
			error_radians = torch.acos(cosines)
			# increase by the number of directions within the threshold
			within_count += error_radians[error_radians < self.threshold_radians].numel()
			# increase by the number of all directions
			total_count += error_radians.numel()

		return torch.tensor(within_count / total_count)


class InverseAngularError:
	def __init__(self, **kwargs):
		pass

	def __call__(self, inputs, targets, **kwargs):
		assert isinstance(inputs, list)
		if len(inputs) == 1:
			targets = [targets]
		assert len(inputs) == len(targets)

		total_error = 0
		for input, target in zip(inputs, targets):
			# normalize and multiply by the stability_coeff in order to prevent NaN results from torch.acos
			stability_coeff = 0.999999
			input = input / torch.norm(input, p=2, dim=1).detach().clamp(min=1e-8) * stability_coeff
			target = target / torch.norm(target, p=2, dim=1).detach().clamp(min=1e-8) * stability_coeff
			# compute cosine map
			cosines = (input * target).sum(dim=1)
			error_radians = torch.acos(cosines)
			total_error += error_radians.sum()

		return torch.tensor(1. / total_error)


def get_evaluation_metric(config):
	"""
	Returns the evaluation metric function based on provided configuration
	:param config: (dict) a top level configuration object containing the 'eval_metric' key
	:return: an instance of the evaluation metric
	"""

	def _metric_class(class_name):
		m = importlib.import_module('pytorch3dunet.unet3d.metrics')
		clazz = getattr(m, class_name)
		return clazz

	assert 'eval_metric' in config, 'Could not find evaluation metric configuration'
	metric_config = config['eval_metric']
	metric_class = _metric_class(metric_config['name'])
	return metric_class(**metric_config)
