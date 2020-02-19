from torch import nn as nn
import numpy as np
import torch
from custom_pt_layers import MaskedLinear, Flatten
from enum import Enum
import os
import torch
import pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
Sparsification mode which can be :
- Mask to remove non-critical neurons
- Random to randomly prune neurons
- Critical to prune critical neurons
"""


class Mode(Enum):
    MASK = 0
    Random = 1
    CRITICAL = 2


def bound_propagation(model, initial_bound):
    """propagate initial upper and lower bound batch of data through the model to get upper/lower for each layer

    Arguments:
        model {models.Model} -- trained model that will be represented by the solver
        initial_bound {tuple} -- upper and lower bound of each input data point to the solver

    Returns:
        list -- list of upper and lower bounds for each layer
    """
    l, u = initial_bound
    bounds = []

    for layer in model:
        if isinstance(layer, Flatten):
            l_ = Flatten()(l)
            u_ = Flatten()(u)
        elif isinstance(layer, nn.Linear):
            l_ = (layer.weight.clamp(min=0) @ l.t() + layer.weight.clamp(max=0) @ u.t()
                  + layer.bias[:, None]).t()
            u_ = (layer.weight.clamp(min=0) @ u.t() + layer.weight.clamp(max=0) @ l.t()
                  + layer.bias[:, None]).t()
        elif isinstance(layer, nn.Conv2d):
            l_ = (nn.functional.conv2d(l, layer.weight.clamp(min=0), bias=None,
                                       stride=layer.stride, padding=layer.padding,
                                       dilation=layer.dilation, groups=layer.groups) +
                  nn.functional.conv2d(u, layer.weight.clamp(max=0), bias=None,
                                       stride=layer.stride, padding=layer.padding,
                                       dilation=layer.dilation, groups=layer.groups) +
                  layer.bias[None, :, None, None])

            u_ = (nn.functional.conv2d(u, layer.weight.clamp(min=0), bias=None,
                                       stride=layer.stride, padding=layer.padding,
                                       dilation=layer.dilation, groups=layer.groups) +
                  nn.functional.conv2d(l, layer.weight.clamp(max=0), bias=None,
                                       stride=layer.stride, padding=layer.padding,
                                       dilation=layer.dilation, groups=layer.groups) +
                  layer.bias[None, :, None, None])
        elif isinstance(layer, nn.MaxPool2d) or isinstance(layer, nn.AvgPool2d) or isinstance(layer, nn.AdaptiveAvgPool2d):
            l_ = layer(l)
            u_ = layer(u)
        elif isinstance(layer, nn.ReLU):
            l_ = l.clamp(min=0)
            u_ = u.clamp(min=0)

        bounds.append((l_.detach().cpu().numpy(), u_.detach().cpu().numpy()))
        l, u = l_, u_
    return bounds


def test_batch(model, data, target, criterion=None):
    """used to compute accuracy and loss of a batch of data points

    Arguments:
        model {Model} -- model to be tested
        data {tensor} -- batch of data points to be tested
        target {tensor} -- target labels of the input batch

    Keyword Arguments:
        criterion {nn} -- torch loss to compute model's loss on input batch (default: {None})

    Returns:
        tuple -- returns number of correct labels and loss computed based on given criterion
    """
    data_loss = -1
    correct = -1
    with torch.no_grad():
        output = model(data)
        if criterion is not None:
            data_loss = criterion(output, target).item()
        # get the index of the max log-probability
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
    return correct, data_loss


def test(model, data_loader, criterion):
    """Test results of a model on a given data loader

    Arguments:
        model {models.Model} -- pytorch model to be tested
        data_loader {torch.dataloader} -- data loader that will be tested
        criterion {nn} -- torch loss to evaluate the data

    Returns:
        tuple -- data loss using given criterion and data accuracy
    """
    model.eval()
    data_loss = 0
    correct = 0
    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        batch_result = test_batch(model, data, target, criterion)
        correct += batch_result[0]
        data_loss += batch_result[1]
    data_loss /= len(data_loader.dataset)
    data_accuracy = 100. * correct / len(data_loader.dataset)
    return data_loss, data_accuracy


def weights_init(m):
    """weight initialization for pytorch model

    Arguments:
        m {nn.module} -- module to be initialized
    """
    classname = m.__class__.__name__.lower()
    if classname.find('conv2d') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('batchnorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('maskedlinear') != -1:
        nn.init.kaiming_normal_(m.linear.weight.data)
        nn.init.constant_(m.linear.bias.data, 0)
    elif classname.find('linear') != -1:
        nn.init.kaiming_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)


def get_storage_dir(storage_parent_dir, config, model_name, exp_indx=0):
    """creates the storage directory path based on current experiment configuration

    Arguments:
        storage_parent_dir {string} -- path of the parent directory used to save experiments
        config {args} -- configuration holding argument's values for this experiment
        model_name {string} -- name of the model used in this experiment

    Keyword Arguments:
        exp_indx {int} -- for each setup we have multiple experiments with different init this to denote experiment number within the same config (default: {0})

    Returns:
        string -- save path
    """
    return os.path.join(storage_parent_dir, model_name, 'lr_'+str(
        config.learning_rate), 'epoch_'+str(config.epochs), 'dataset_'+str(config.dataset), 'optimizer_'+str(config.optimizer), 'exp_'+str(exp_indx))


def save_pickle(out_file_path, data_points):
    """pickle a set of data points

    Arguments:
        out_file_path {string} -- file path used to save the pickle dump
        data_points {object} -- data to be saved in pickle
    """
    with open(out_file_path, 'wb') as handle:
        pickle.dump(data_points, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_current_lr(optimizer):
    """returns the current learning rate used by the optimizer

    Arguments:
        optimizer {torch.optim} -- optimizer used during training

    Returns:
        float -- learning rate used
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def adjust_learning_rate(optimizer, epoch, lr_multiplier=0.5, epoch_checkpoints=[100, 150, 200]):
    """automatic adjustment of learning rate used for Cifar10 experiments only

    Arguments:
        optimizer {torch.optim} -- torch optimizer used during training
        epoch {int} -- current epoch number

    Keyword Arguments:
        lr_multiplier {float} -- learning rate scaler (default: {0.5})
        epoch_checkpoints {list} -- epoch checkpoint used to adjust learning rate (default: {[100, 150, 200]})

    Returns:
        [type] -- [description]
    """
    lr = get_current_lr(optimizer)
    if epoch in epoch_checkpoints:
        lr *= lr_multiplier
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
