import os
import glob
import pickle
import argparse
import numpy as np
import h5py


parser = argparse.ArgumentParser()
parser.add_argument('--dirpath', default='', help='path to ground truth pickle files')
parser.add_argument('--prefix', default='', help='path to EGFP data files')

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

    def get_intensities(self, prefix_path, coordinates_dir):
        """
        Get the intensity variation for each source
        across time steps.
        prefix_path: path to EGFP files
        """

        intensity_dir = {}
        for key, val in coordinates_dir.items():
            for num in range(18):
                raw = h5py.File(prefix_path + 'EGFP_' + str(num) + '_new.h5', 'r')['raw']
                if key not in intensity_dir:
                    intensity_dir[key] = []
                    intensity_dir[key].append(raw[0][int(val[num][2])-12, int(val[num][1])-426,\
                     int(val[num][0])-426])
                else:
                    intensity_dir[key].append(raw[0][int(val[num][2])-12, int(val[num][1])-426,\
                     int(val[num][0])-426])

        return intensity_dir

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
        """
        Record the movement at each timestep for all sources
        """

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

    @staticmethod
    def show_displacement(deviation_dir):
        """
        Method to display displacement for each source
        """

        for filenum in range(1, 41):
            if filenum in range(1, 10):
                print('Pair' + str(filenum) + '_1 ' + str(deviation_dir[' Pair0' + str(filenum) + '_1']))
                try:
                    print('Pair' + str(filenum) + '_2 ' + str(deviation_dir[' Pair0' + str(filenum) + '_2']))
                except:
                    pass
            else:
                print('Pair' + str(filenum) + '_1 ' + str(deviation_dir[' Pair' + str(filenum) + '_1']))
                print('Pair' + str(filenum) + '_2 ' + str(deviation_dir[' Pair' + str(filenum) + '_2']))

if __name__ == '__main__':
    args = parser.parse_args()
    dirpath = args.dirpath
    prefix = args.prefix

    src_dis = SourceDisplacement()
    # call required methods here after
