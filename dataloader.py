from src.dataset import load_dataset
from torch.utils.data import TensorDataset

batch_size = 2

def data_loader(ds_input, val_ds_input, ds_target, val_ds_target, batch_size):
	#creating a tensordataset
	train_ds = TensorDataset(ds_input, ds_target)
	valid_ds = TensorDataset(val_ds_input, val_ds_target)

	# print(ds_input.shape)

	#provide a dataloader
	train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
	valid_loader = DataLoader(valid_ds, batch_size=batch_size)

	# print(len(train_loader))
	# print(len(valid_loader))
	return train_loader, valid_loader

