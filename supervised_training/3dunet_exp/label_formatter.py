"""Annotations for the segmentation
with 3D Unet (supervised)
"""

import argparse
import pickle
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('annotation_csv', default='', help="Path to annotation csv file")

# embldata/Zygote112.csv'

def simplify_annotations(annotation_csv):
    """Get relevant info
    from the annotations file"""

    annotation_file = pd.read_csv(annotation_csv)
    tags = annotation_file['Tags']

    tags_list = list(tags)
    Xpx = list(annotation_file['X (px), Center of Mass (Intensities) #1'])
    Ypx = list(annotation_file['Y (px), Center of Mass (Intensities) #1'])
    Zpx = list(annotation_file['Z (px), Center of Mass (Intensities) #1'])

    pair_num_list = []

    for each_tag in tags_list:
        pair_num = each_tag.split('.')[2].strip()
        pair_num_list.append(pair_num)

    return pair_num_list, Xpx, Ypx, Zpx

def pickle_creator(pair_num_list, Xpx, Ypx, Zpx):
    """Creates annotations per
    file -> total 18 files.
    Jumbled annotations, thus 
    getting individual source
    from the csv.
    Annotation are similar for
    EGFP and mcherry data files.
	"""

    dict_list = []
    for filenum in range(18):

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

    for k, dicts in enumerate(dict_list):
        file_name = 'file' + str(k) + '.pickle'
        with open(file_name, 'wb') as handle:
            pickle.dump(dicts, handle)


if __name__ == '__main__':

    args = parser.parse_args()
    annotations_path = args.annotation_csv

    num_list, Xpx, Ypx, Zpx = simplify_annotations(annotations_path)
    pickle_creator(num_list, Xpx, Ypx, Zpx)
