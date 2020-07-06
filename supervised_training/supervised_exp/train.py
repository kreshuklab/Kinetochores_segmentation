# Initialize model, loss and optimizer

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset, DataLoader

import argparse

from GSmoothing.GSmoothing import GaussianSmoothing
from loader.Loader import dloader
from lossfunc.lossfunc import RMSLELoss
from model.model import UnetModel
from preproc.preproc import preproc
from std.standardize import standardize

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir_path', default='', help='provide path to dataset')
parser.add_argument('--batch_size', default=1)
parser.add_argument('--num-epochs', default=5, help='Number of epochs')
parser.add_argument('--lr', default=0.0001, help='Learning rate for training')

# parser.add_argument('--optimizer', default='')

#scheduler = StepLR(optimizer, step_size=2, gamma=0.7)

#scheduler = StepLR(optimizer, step_size=2, gamma=0.92)

def plot_figs(gtruth, output):
    fig = plt.figure()

    tar_grid = torchvision.utils.make_grid(gtruth[:,:,15:16].squeeze(2))
    #print(tar_grid)

    fig.add_subplot(1, 2, 1)
    plt.imshow(tar_grid.permute(1,2,0), cmap='Blues')#, vmin=0., vmax=1.)

    out_grid = torchvision.utils.make_grid(output[:,:,15:16].squeeze(2))
    #print(out_grid.shape)

    fig.add_subplot(1, 2, 2)
    plt.imshow(out_grid.permute(1,2,0), cmap='Blues')#, vmin=0., vmax=1.)

    return fig


def train(model, train_loader, valid_loader, num_epochs, smoothing, criterion, optimizer, device):
    train_loss = 0.0
    valid_loss = 0.0

    device = torch.device('cuda')

    for epoch in range(1, num_epochs+1):
        model.train()
        t_loss = 0.0

        for ip, target in train_loader:

            gauss_target = smoothing(target)
            ip, gauss_target = ip.to(device), gauss_target.to(device)

            pred = model(ip)

            gauss_pred = smoothing(pred.cpu())
            #gauss_target = smoothing(target.cpu)
            gauss_pred = gauss_pred.to(device)
            #Loss of generated vol w.r.t original vol
            loss_temp = criterion(gauss_pred, gauss_target)

            loss_temp.backward()
            optimizer.step()
            optimizer.zero_grad()

            t_loss += loss_temp.item()

        train_loss = t_loss / len(ip)   # batch_size

        #scheduler.step()
        writer.add_scalar('train_loss', train_loss)

        if epoch % 1 == 0:
            print('Epoch {} of {}, Train Loss: {:.3f}'.format(
                epoch, num_epochs, train_loss / (len(train_loader))))

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for xval, yval in valid_loader:
                gauss_yval = smoothing(yval)
                xval, gauss_yval = xval.to(device), gauss_yval.to(device)

                op = model(xval)

                gauss_xval = smoothing(op.cpu())
                gauss_xval = gauss_xval.to(device)

                loss = criterion(gauss_xval, gauss_yval)

                if epoch % 1 == 0:
                    writer.add_figure('Panel 1: pred vs actual at epoch{}'.format(epoch), plot_figs(gauss_xval.cpu(), gauss_yval.cpu()))

                v_loss += loss.item()

            valid_loss = v_loss / len(xval)

            writer.add_scalar('val_loss', valid_loss)
            # if epoch == 1:
            #     loss_for_step = valid_loss
            # else:
            #     if valid_loss < loss_for_step:
            #         loss_for_step = valid_loss
            #         scheduler.step()

        if epoch % 1 == 0:
            print('Epoch {}, Valid Loss: {:.3f}'.format(
                epoch, valid_loss / len(valid_loader)))

    return train_loss, valid_loss

if __name__ == '__main__':

    args = parser.parse_args()
    num_epochs = args.num_epochs
    learning_rate = args.lr
    batch_size = args.batch_size

    dir_path = args.data_dir_path
    train_dir_path = dir_path + 'train/'
    valid_dir_path = dir_path + 'val/'

    data_input, data_target, val_data_input, val_data_target = preproc.getdata(train_dir_path, valid_dir_path)

    ds_input, ds_target, val_ds_input, val_ds_target = preproc.get_tensors(data_input, data_target, val_data_input, val_data_target)

    ds_input, val_ds_input = standardize.standardize_ds(ds_input, val_ds_input)

    train_loader, valid_loader = dloader.load_data('', batch_size, ds_input, ds_target, val_ds_input, val_ds_target)

    smoothing = GaussianSmoothing(1, 30, 11, 3)

    criterion = RMSLELoss()
    model = UnetModel(in_channels=1, out_channels=1)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    writer = SummaryWriter('sup_loss/exp1')

    device = torch.device('cuda')

    model.to(device)

    train(model, train_loader, valid_loader, num_epochs, smoothing, criterion, optimizer, device)
