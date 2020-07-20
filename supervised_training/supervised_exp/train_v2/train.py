# Initialize model, loss and optimizer

import os
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

from torch.optim.lr_scheduler import StepLR

from skimage.filters import gaussian

import argparse
import h5py

from GSmoothing.GSmoothing import GaussianSmoothing
from loader.Loader import dloader
from lossfunc.lossfunc import RMSLELoss
from model.model import UnetModel
from preproc.preproc import preproc 
from std.standardize import standardize
from lossfunc.wce import weighted_categorical_crossentropy
from diceloss.dice import DiceL

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir_path', default='', help='provide path to dataset')
parser.add_argument('--batch_size', default=1)
parser.add_argument('--num-epochs', default=12, help='Number of epochs')
parser.add_argument('--lr', default=0.0001, help='Learning rate for training')
# parser.add_argument('--optimizer', default='')


#scheduler = StepLR(optimizer, step_size=2, gamma=0.7)

#scheduler = StepLR(optimizer, step_size=2, gamma=0.92)

model_dir = 'models/'

"""def plot_figs(gtruth, output):
    fig = plt.figure()
    
    tar_grid = torchvision.utils.make_grid(gtruth[:,:,20:21].squeeze(2))
    #print(tar_grid.shape)

    fig.add_subplot(1, 2, 1)
    #print(tar_grid.permute(1,2,0).shape)
    #plt.imshow(tar_grid)#.permute(1,2,0), cmap='Blues')#, vmin=0., vmax=1.)
    #plt.imshow(gtruth[:][0][0][24].numpy())
    
    #out_grid = torchvision.utils.make_grid(output[:,:,20:21].squeeze(2))
    #print(out_grid.shape)
    
    #fig.add_subplot(1, 2, 2)
    #print(out_grid.permute(1,2,0).shape)
    #plt.imshow(out_grid)#.permute(1,2,0), cmap='Blues')#, vmin=0., vmax=1.)
    #plt.imshow(output[:][0][0][24].numpy())
    tar_grid = torchvision.utils.make_grid(gtruth[:,:,20:21].squeeze(2), nrow=2)
    plt.imshow(tar_grid.permute(1,2,0))
    fig.add_subplot(1,2,1)

    out_grid = torchvision.utils.make_grid(output[:,:,20:21].squeeze(2), nrow=2)
    plt.imshow(out_grid.permute(1,2,0))
    fig.add_subplot(1,2,2)

    #com_grid = [tar_grid, out_grid]
    #ax = fig.add_subplot(1, 2, 1)
    
    #x = torchvision.utils.make_grid(com_grid)
    #ax.imshow(x.numpy().transpose(1,2,0))

    return fig"""


def save_hdf5(stage, epoch, pred, target):
    pred = pred.detach().numpy()
    target = target.detach().numpy()
    if stage == 'pre':
        hfile = h5py.File('pre-gaussian-{}-extemp2.h5'.format(str(epoch)), 'w')
    else:
        hfile = h5py.File('post-gaussian-{}-extemp2.h5'.format(str(epoch)), 'w')
    hfile.create_dataset('pred', data=pred)
    hfile.create_dataset('target', data=target)
    hfile.close()

def apply_gaussian(vol, sigma):
    return gaussian(vol, sigma=sigma)

def WCELoss(weights, input, target):
    loss = nn.CrossEntropyLoss(weight=weights, reduction='none')
    #input = input.squeeze(1)
    #target = target.squeeze(1)
    #loss_w = loss(input, target)
    #loss_w = loss_w.sum() / weights[target].sum()
    return loss_w

def WMSE(input, target, weights):
    out = (input - target) ** 2
    out = out * weights.expand_as(out)
    loss = out.sum(0)
    return loss

#def dice(pred, target):
#    numerator = 2 * torch.sum(pred * target)
#    denominator = torch.sum(pred + target)
#    return 1 - (numerator + 1) / (denominator + 1)


def train(model, train_loader, valid_loader, num_epochs, smoothing, criterion, optimizer, device):
    train_loss = 0.0
    valid_loss = 0.0
    
    device = torch.device('cuda')
    #weights = torch.FloatTensor([0.01, 1.]).cuda()

    scheduler = StepLR(optimizer, step_size=2, gamma=0.92)

    for epoch in range(1, num_epochs+1):
        model.train()
        t_loss = 0.0

        for ip, target in train_loader:

            #gauss_target = smoothing(target)
            ip, target = ip.to(device), target.to(device)

            pred = model(ip)
            
            if epoch % 2 == 0:
                torch.save(model.state_dict(), os.path.join(model_dir, 'mon-extemp2-pregaussian-epoch-{}.pth'.format(epoch)))

            if epoch % 2 == 0:
                save_hdf5('pre', epoch, pred.cpu(), target.cpu())

            #g_pred = F.pad(pred.cpu(), (2, 2, 2, 2, 2), mode='reflect')
            #gauss_pred = smoothing(g_pred)
            
            gauss_pred = smoothing(pred.cpu())

            #gauss_pred = apply_gaussian(pred.cpu().numpy(), sigma=2)
            #gauss_pred = gaussian(pred.cpu().detach().numpy(), sigma=2)
            #g_tar = F.pad(target.cpu(), (2, 2, 2, 2, 2), mode='reflect')
            #gauss_target = smoothing(g_tar)
            
            gauss_target = smoothing(target.cpu())

            #gauss_target = apply_gaussian(target.cpu().numpy(), sigma=2)
            #gauss_target = gaussian(target.cpu().detach().numpy(), sigma=2)
            
            if epoch % 2 == 0:
                save_hdf5('post', epoch, gauss_pred, gauss_target)

            #gauss_pred, gauss_target = torch.tensor(gauss_pred).to(device), torch.tensor(gauss_target).to(device)
            
            gauss_pred, gauss_target = gauss_pred.to(device), gauss_target.to(device)
            
            #Loss of generated vol w.r.t original vol
            #print(gauss_pred.shape, gauss_target.shape)
            #loss_temp = criterion(gauss_pred, gauss_target)
            #loss_temp = WCELoss(weights, gauss_pred, gauss_target)
            #loss_temp = WMSE(gauss_pred, gauss_target, weights)
            #loss_temp = DiceLoss(gauss_pred, gauss_target)
            #loss_temp = dice(gauss_pred, gauss_target)
            
            loss_temp = criterion(gauss_pred, gauss_target)

            loss_temp.backward()
            optimizer.step()
            optimizer.zero_grad()

            if epoch%2 == 0:
                torch.save(model.state_dict(), os.path.join('extemp2-post-gaussian-epoch-{}.pth'.format(epoch)))

            t_loss += loss_temp.item()

        train_loss = t_loss / len(ip)   # batch_size
        
        #scheduler.step()
        writer.add_scalar('train_loss', train_loss, global_step=epoch)

        #writer.add_graph(model, ip)

        #writer.add_embedding(ip.view(-1, 1*48*128*128), label_img=ip.unsqueeze(1))

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

                #loss = criterion(gauss_xval, gauss_yval)
                #loss = WCELoss(weights, gauss_xval, gauss_yval)
                #loss = WMSE(gauss_xval, gauss_yval, weights)
                #loss = DiceLoss(gauss_xval, gauss_yval)
                loss = criterion(gauss_xval, gauss_yval)

                #if epoch % 1 == 0:
                #grid = torchvision.utils.make_grid(gauss_xval[:,:,20:21].squeeze(2))
                #f_prob, ax_prob = plt.subplots()
                #plt.plot(grid)
                #writer.add_figure('Pred', f_prob, step=epoch)
                #writer.add_image('pred', grid, 0)
                #writer.add_image('Prediction at epoch{}'.format(epoch), grid, global_step=epoch, dataformats='CHW')
                #writer.add_image('Target_extemp', gauss_yval[:,:,20:21].squeeze(2), dataformats='NCHW')
                
                #writer.add_figure('Panel 1: pred vs actual at epoch{}'.format(epoch), plot_figs(gauss_yval.cpu(), gauss_xval.cpu()))

                v_loss += loss.item()

            valid_loss = v_loss / len(xval)
            
            writer.add_scalar('val_loss', valid_loss, global_step=epoch)
        if epoch == 1:
            loss_for_step = valid_loss
        else:
            if valid_loss < loss_for_step:
                loss_for_step = valid_loss
                scheduler.step()

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

    smoothing = GaussianSmoothing(1, 5, 1, 3)

    criterion = RMSLELoss()
    #criterion = DiceL()
    model = UnetModel(in_channels=1, out_channels=1)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    writer = SummaryWriter('tboard/mon-extemp2')

    device = torch.device('cuda')

    model.to(device)

    train(model, train_loader, valid_loader, num_epochs, smoothing, criterion, optimizer, device)
