import os
import glob
import pickle
import numpy as np
import pandas as pd


class SourceDisplacement:

	def __init__(self):
		pass

	def get_trajectories(self, dirpath):
		"""
		Prepare the trajectories for each source
		dirpath: path to directory with all the ground truth pickle files
		"""

		#gt_file_list = os.listdir('/home/samudre/embldata/pickles/pickles/')
		gt_file_list = os.listdir(dirpath)

		coordinates_dir = {}

		for each_pkl in gt_file_list:
			#with open('/home/samudre/embldata/pickles/pickles/' + each, 'rb') as hn:
			with open(dirpath + each_pkl, 'rb') as handle:	
				gt_file = pickle.load(handle)
			for key, val in gt_file.items():
				if key not in coordinates_dir.keys():
					coordinates_dir[key] = []
					coordinates_dir[key].append(val)
				else:
					coordinates_dir[key].append(val)

		return coordinates_dir

	@staticmethod
	def get_displacement(coordinates_dir):
		"""
		Prepare coordinates deviation directory
		coordinates_dir: directory with all source trajectories
		"""

		xlist = []
		ylist = []
		zlist = []

		deviation_dir = {}

		for key, val in coordinates_dir.items():
			for coord_vals in val:
				xlist.append(coord_vals[0])
				ylist.append(coord_vals[1])
				zlist.append(coord_vals[2])
			deviation_dir[key] = (np.std(xlist), np.std(ylist), np.std(zlist))

		return deviation_dir

	@staticmethod
	def get_movement(coordinates_dir):

		x_movement = {}
		y_movement = {}
		z_movement = {}

		for key, val in coordinates_dir.items():
			for coord_vals in val:
				if key not in x_movement.keys():
					x_movement[key] = []
					x_movement[key].append(coord_vals[0])
					y_movement[key] = []
					y_movement[key].append(coord_vals[1])
					z_movement[key] = []
					z_movement[key].append(coord_vals[2])
				else:
					x_movement[key].append(coord_vals[0])
					y_movement[key].append(coord_vals[1])
					z_movement[key].append(coord_vals[2])

		return x_movement, y_movement, z_movement

	@staticmethod
	def get_neighborhood(deviation_dir):
		"""
		Get the mean displacement in x, y and z
		deviation_dir: std deviation for all the sources
		"""
		
		xdisp = []
		ydisp = []
		zdisp = []

		for key, val in deviation_dir.items():
			xdisp.append(val[0])
			ydisp.append(val[1])
			zdisp.append(val[2])

		return np.mean(xdisp), np.mean(ydisp), np.mean(zdisp)



	def show_displacement(self, deviation_dir):
		"""
		Method to display displacement for each source
		"""
		for filenum in range(1,41):
			if filenum in range(1,10):
				print('Pair' + str(filenum) + '_1 ' + str(deviation_dir[' Pair0' + str(filenum) + '_1']))
				try:
					print('Pair' + str(filenum) + '_2 ' + str(deviation_dir[' Pair0' + str(filenum) + '_2']))
				except:
					pass
			else:
				print('Pair' + str(filenum) + '_1 ' + str(deviation_dir[' Pair' + str(filenum) + '_1']))
				print('Pair' + str(filenum) + '_2 ' + str(deviation_dir[' Pair' + str(filenum) + '_2']))
