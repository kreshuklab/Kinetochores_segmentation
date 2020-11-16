import os
import tempfile

import h5py
import numpy as np

from pytorch3dunet.datasets.utils import ConfigDataset
from pytorch3dunet.unet3d.utils import get_logger

logger = get_logger('EGFPDataset')


class EGFPDataset(ConfigDataset):
    def __init__(self, file_path, internal_path, z_slice_count, target_slice_index, **kwargs):
        with h5py.File(file_path, 'r') as f:
            self.raw = f[internal_path][...]

        self.z_slice_count = z_slice_count
        self.target_slice_index = target_slice_index

        assert self.raw.ndim == 4
        # assumes ZYXC axis order
        assert self.raw.shape[0] >= z_slice_count

        assert target_slice_index < z_slice_count

    def __getitem__(self, index):
        if self.target_slice_index == 0:
            # 1st z-slice is the target
            # return rank 4 tensor always: ZYXC
            return self.raw[index + 1:index + self.z_slice_count], \
                   np.expand_dims(self.raw[index + self.target_slice_index], axis=0)
        elif self.target_slice_index == self.z_slice_count - 1:
            # last z-slice is the target
            return self.raw[index:index + self.target_slice_index], \
                   np.expand_dims(self.raw[index + self.target_slice_index], axis=0)
        else:
            lower_sub_volume = self.raw[index:index + self.target_slice_index]
            upper_sub_volume = self.raw[index + self.target_slice_index + 1:index + self.z_slice_count]

            input = np.concatenate([lower_sub_volume, upper_sub_volume])
            target = np.expand_dims(self.raw[index + self.target_slice_index], axis=0)

            return input, target

    def __len__(self):
        return self.raw.shape[0] - self.z_slice_count + 1

    @classmethod
    def create_datasets(cls, dataset_config, phase):
        internal_path = dataset_config.get('internal_path', 'exported_data')
        z_slice_count = dataset_config.get('z_slice_count', 3)
        target_slice_index = dataset_config.get('target_slice_index', z_slice_count - 1)
        phase_config = dataset_config[phase]
        # load files to process
        file_paths = phase_config['file_paths']

        return [cls(file_path, internal_path, z_slice_count, target_slice_index) for file_path in file_paths]

# if __name__ == '__main__':
#     with tempfile.TemporaryDirectory() as d:
#         file_path = os.path.join(d, 'test.h5')
#         with h5py.File(file_path, 'w') as f:
#             f.create_dataset('exported_data', data=np.random.rand(60, 100, 100, 1), compression='gzip')

#         ds = EGFPDataset(file_path, 'exported_data', 3, 0)

#         print('>> ds lenght: ', len(ds))

#         for input, target in ds:
#             print(f'input shape: {input.shape}, target shape: {target.shape}')
