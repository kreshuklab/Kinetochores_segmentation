import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import logging
import os
import numpy as np

import utils
from torch.utils.data import DataLoader
from preprocessing import data_reshape, data_standardize
from dataloader import data_loader
from model.model import *
from model.loss_fn import RMSLELoss


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='',
                    help='directory containing parameter config file')


def train(model, data_loader, optimizer, loss_fn, params, device):
    """Train the model for num of epochs

    Args:
        model: The 3D UNet (torch.nn.Module)
        dataloader: Object to fetch the training data (torch.utils.data.DataLoader)
        optimizer: Optimizer object for weight updates (torch.optim)
        loss_fn: Loss function 
        num_epochs: number of epochs for training (int)
        device: The gpu device, if available ('cuda') 
    
    """

    model.train()
    loss_avg = utils.RunningAverage()

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if params.cuda:
            input_batch, target_batch = input_batch.to(device),\
            target_batch.to(device)

        output_batch = model(input_batch)
        loss = loss_fn(output_batch, input_batch)

        optimizer.zero_grad()

        loss_temp.backward()
        optimizer.step()

        # update the average loss
        loss_avg.update(loss.item())

        # if epoch % 4 == 0:
        #     print('Epoch {} of {}, Train Loss: {:.3f}'.format(
        #         epoch, num_epochs, train_loss / (len(train_loader))))


def evaluate(model, loss_fn, dataloader, metrics, params):
    """Evaluate the model on `num_steps` batches.
    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = []

    # compute metrics over the dataset
    for data_batch, target_batch in dataloader:

        # move to GPU if available
        if params.cuda:
            data_batch, labels_batch = data_batch.cuda(),\
            target_batch.cuda()
        
        # compute model output
        output_batch = model(data_batch)
        loss = loss_fn(output_batch, target_batch)

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output_batch = output_batch.data.cpu() #.numpy()
        labels_batch = labels_batch.data.cpu() #.numpy()

        summ.append(loss.item())


def train_and_eval(model, train_loader, valid_loader, optimizer, loss_fn, params, model_dir, restore_file=None):
    """Train model and evaluate on validation set per epoch.
    
    Args:
        model:
        train_loader:
        valid_loader:
        optimizer:
        loss_fn:
        model_dir:
        restore_file:
    """
    
    ###### reload weights from restore_file if specified
    # if restore_file is not None:
    #     restore_path = os.path.join(
    #         args.model_dir, args.restore_file + '.pth.tar')
    #     logging.info("Restoring parameters from {}".format(restore_path))
    #     utils.load_checkpoint(restore_path, model, optimizer)

    best_val_loss = 1e5

    for epoch in range(params.num_epochs):
        logging.info("Epoch {}/{}".format(epoch+1, params.num_epochs))

        train(model, train_loader, optimizer, loss_fn, params)

        val_loss = evaluate(model, valid_loader, loss_fn, params)

        is_best = val_loss <= best_val_loss

        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                                'optim_dict': optimizer.state_dict()},
                                is_best=is_best,
                                checkpoint=model_dir)

        # If best_eval, use best_save_path
        if is_best:
            logging.info(" Best loss")
            best_val_loss = val_loss

            best_json_path = os.path.join(
                model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_loss, best_json_path)

        last_json_path = os.path.join(model_dir, "val_last_weights.json")
        utils.save_dict_to_json(val_loss, last_json_path)

if __name__ == '__main__':

    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path, "No json config found at {}". format(json_path))

    params = utils.Params(json_path)

    params.cuda = torch.cuda.is_available()

    torch.manual_seed(42)

    if params.cuda:
        torch.cuda.manual_seed(42)

    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    logging.info("Loading data")

    ##### get dataloaders

    # dataloaders = data_loader.fetch_dataloader(
    #     ['train', 'val'], args.data_dir, params)
    # train_dl = dataloaders['train']
    # val_dl = dataloaders['val']

    data, val_data = load_dataset(args.data_dir)

    ds_target, val_ds_target = data_reshape(data, val_data)
    ds_input, val_ds_input, ds_target, val_ds_target = data_standardize(ds_target,\
     val_ds_target)

    train_loader, valid_loader = data_loader(ds_input, val_ds_input,\
        ds_target, val_ds_target)

    # define model and optimizer

    model = model.Net(params).cuda() if params.cuda else model.Net(params)
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    ## TODO: add scheduler

    loss_fn = model.loss_fn

    #Train model
    logging.info("Training for {} epochs".format(params.num_epochs))

    train_and_eval(model, train_dl, val_dl, optimizer, loss_fn, params, args.model_dir, args.restore_file)
