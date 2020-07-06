"""
Loader module
"""

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

class dloader(object):
    def __init__(self):
        pass

    def load_data(self, batch_size, ds_input, ds_target, val_ds_input, val_ds_target):
        """
        bs: batch size (int)
        ds_input: input dataset (Tensor)
        ds_target: target dataset (Tensor)
        val_ds_input: validation input dataset (Tensor)
        val_ds_target: validation target dataset (Tensor)
        """

        #creating a tensordataset
        train_ds = TensorDataset(ds_input, ds_target)
        valid_ds = TensorDataset(val_ds_input, val_ds_target)

        #provide a dataloader
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_ds, batch_size=batch_size)

        return train_loader, valid_loader
