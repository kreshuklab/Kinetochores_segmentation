import pandas as pd
import numpy as np
from scipy.spatial import distance
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--csv', default='', help='Path to csv with predictions and GT labels')


def SpatialDistanceEval(csvfile):
	"""
	csvfile: csv with peaks and labels (GT)
	"""

	# Read file as dataframe
	data_file = pd.read_csv(csvfile)

	# Ground truth coordinates
	GT_x = list(data_file['Gtx'])
	GT_y = list(data_file['Gty'])
	GT_z = list(data_file['Gtz'])

	# Prediction peak coordinates
	Pred_x = list(data_file['PredX'])
	Pred_y = list(data_file['PredY'])
	Pred_z = list(data_file['PredZ'])

	# Dict to store matching pairs
	eval_dict = {}

	# Check peaks for each label coordinate
	for x,y,z in zip(GT_x, GT_y, GT_z):
		eval_dict[(x,y,z)] = []
		for px, py, pz in zip(Pred_x, Pred_y, Pred_z):
			
			# Take the standardized euclidean distance
			std_euc_distance = distance.seuclidean([x,y,z], [px, py, pz], [1.3,1.3,3.0])
			
			# Take the euclidean distance
			#dist = distance.euclidean([x,y,z], [px, py, pz])

			#Check for threshold
			if std_euc_distance <= 8.0:
				eval_dict[(x,y,z)].append((px,py,pz, std_euc_distance))

	# Count the unmatched label coordinates
	unmatched_count = 0
	for each_label in eval_dict.values():
		if each_label == []:
			unmatched_count += 1

	return unmatched_count, eval_dict


if __name__ == '__main__':
	args = parser.parse_args()
	csvfile = args.csv

	unmatched_count, eval_dict = SpatialDistanceEval(csvfile)

	for each, val in eval_dict.items():
		print(each, val)