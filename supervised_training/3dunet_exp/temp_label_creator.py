import h5py
import glob
import pickle
import numpy as np

# flist = glob.glob('/home/samudre/embldata/EGFP/labeldir/*')

# for each in flist:
#     target = np.zeros((1, 48, 128, 128))
    
#     with open(each, 'rb') as new_pkl:
#         labels = pickle.load(new_pkl)
    
#     for k, v in labels.items():
#         a, b, c = int(v[0] - 426), int(v[1] - 426), int(v[2] - 12)
#         target[0][c, b, a] = 1.0
        
#     tar_h5_name = each.split('/')[-1].split('.')[0] + '_tar.h5'
#     target_file = h5py.File(tar_h5_name, 'w')
#     target_file.create_dataset('labels', data=target)


tarlist = glob.glob('/home/samudre/embldata/EGFP/targets/*')
inplist = glob.glob('/home/samudre/embldata/EGFP/datafiles/*')

