"""Annotations for the segmentation
with 3D Unet (supervised)

simplify_annotations
Extract 79 elements and their 
corresponding coordinates.

Pickle creator
Creates annotations per file -> total 18 files.
Jumbled annotations, thus getting individual
source from the csv. Annotation are similar for
EGFP and mcherry data files.
"""

import argparse
import pickle
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('annotation_csv', default='', help="Path to annotation csv file") # embldata/Zygote112.csv'

def simplify_annotations(annotation_csv):
    """
    annotation_csv: The original annotation file with all the annotations
    """

    annotation_file = pd.read_csv(annotation_csv)
    
    # Extract column names
    tags = annotation_file['Tags']

    # Center of mass coordinates in X, Y, Z
    tags_list = list(tags)
    Xpx = list(annotation_file['X (px), Center of Mass (Intensities) #1'])
    Ypx = list(annotation_file['Y (px), Center of Mass (Intensities) #1'])
    Zpx = list(annotation_file['Z (px), Center of Mass (Intensities) #1'])

    # All the elements are considered individual sources
    # store individual elemnt num as PairN_1 / PairN_2 in pair_num_list
    pair_num_list = []

    for each_tag in tags_list:
        pair_num = each_tag.split('.')[2].strip()
        pair_num_list.append(pair_num)

    return pair_num_list, Xpx, Ypx, Zpx

def pickle_creator(pair_num_list, Xpx, Ypx, Zpx):

    """
    pair_num_list: The individual element list
    Xpx: x coord values
    Ypx: y coord values
    Zpx: z coord values
    """

    # dict_list: 
    timeframes = 18
    dict_list = []

    # for each timestep, create the dict with 
    # key as element num and values as coordinates

    for filenum in range(timeframes):

        temp_dict = {}
        for j, pair in enumerate(pair_num_list):
            if pair in temp_dict.keys():
                continue
            else:
                temp_dict[pair] = [Xpx[j], Ypx[j], Zpx[j]]
                del pair_num_list[j]
                del Xpx[j]
                del Ypx[j]
                del Zpx[j]
        dict_list.append(temp_dict)

    # Store the dicts as pickles per timestep
    for k, dicts in enumerate(dict_list):
        file_name = 'Vol_' + str(k) + '_labels.pickle'
        with open(file_name, 'wb') as handle:
            pickle.dump(dicts, handle)


if __name__ == '__main__':

    args = parser.parse_args()
    annotations_path = args.annotation_csv

    num_list, Xpx, Ypx, Zpx = simplify_annotations(annotations_path)
    pickle_creator(num_list, Xpx, Ypx, Zpx)
