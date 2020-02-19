from models import *
from training import ModelTrain, Logger
import os
import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from dataset import BalancedDataLoader, Caltech256
from training import device, get_storage_dir
import copy
"""Script used to train a model based on input configuration
"""

model_indx_map = {
    0: FullyConnectedBaselineModel,
    1: FullyConnected2Model,
    2: FullyConnected3Model,
    3: FullyConnected4Model,
    4: ConvBaselineModel,
    5: VGG19,
    6: VGG7
}

dataset_map = {
    0: datasets.MNIST,
    1: datasets.FashionMNIST,
    2: datasets.KMNIST,
    3: Caltech256,
    4: datasets.CIFAR10
}

learning_rates = [1e-1, 1e-2, 1e-3, 1e-5]
optimizers = [torch.optim.Adam, torch.optim.SGD, torch.optim.RMSprop]


def process_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', '-m', default=4, type=int)
    parser.add_argument('--epochs', '-e', default=1, type=int)
    parser.add_argument('--batch-size', '-bs', default=32, type=int)
    parser.add_argument('--learning-rate', '-l', default=2, type=int)
    parser.add_argument('--storage-dir', '-sd', default=None)
    parser.add_argument('--other-model-dir', '-omd', default=None)
    # used to train multiple models with different initializations
    parser.add_argument('--num-resets', '-r', default=1, type=int)
    parser.add_argument('--debug', '-d', default=True, type=bool)
    parser.add_argument('--dataset', '-dl', default=0, type=int)
    parser.add_argument('--optimizer', '-op', default=0, type=int)
    parser.add_argument('--retrain', '-rt', action='store_true')
    return parser


def prepare_model(config, input_size=784, n_channels=1, n_output_classes=10):
    if config.storage_dir is None:
        raise Exception(
            'Please provide storage-dir argument to save model data')
    if config.model in model_indx_map:
        model = model_indx_map[config.model](
            input_size=input_size, n_channels=n_channels, n_output_classes=n_output_classes).to(device)
    else:
        raise Exception('Model indx {} is not right'.format(str(config.model)))
    return model


def prepare_model_train(config, input_size, n_channels=1, n_output_classes=10, exp_indx=0, prefix=''):
    model = prepare_model(config, input_size=input_size,
                          n_channels=n_channels, n_output_classes=n_output_classes)
    criterion = torch.nn.CrossEntropyLoss()
    learning_rate = learning_rates[config.learning_rate]
    if optimizers[config.optimizer] is torch.optim.SGD:
        optimizer = optimizers[config.optimizer](
            model.parameters(), lr=learning_rate, nesterov=True, weight_decay=0.0001, momentum=0.9)
    else:
        optimizer = optimizers[config.optimizer](
            model.parameters(), lr=learning_rate)

    storage_parent_dir = config.storage_dir + prefix
    storage_parent_dir = get_storage_dir(
        storage_parent_dir, config, model.name, exp_indx)
    model_train = ModelTrain(
        model, criterion, optimizer, storage_parent_dir, debug=config.debug, input_size=input_size, retrain_masked=config.retrain, other_model_parent_dir=config.other_model_dir)
    return model_train


def prepare_dataset(config, is_color=False):
    if config.dataset in dataset_map:
        transformation_list = [transforms.Resize(
            (32, 32)), transforms.Grayscale(3)]
        if config.model > 4:
            # for deep models
            if config.dataset == 3:  # for calthec dataset
                transformation_list.append(transforms.Resize((224, 224)))

        transformation_list.append(transforms.ToTensor())
        dataset_transform = transforms.Compose(transformation_list)
        test_dataset = dataset_map[config.dataset](
            "../data", train=False, download=True, transform=dataset_transform)
        random_crop_size = next(iter(test_dataset))[0].shape[-1]
        train_transformation = []
        if config.dataset > 2:
            train_transformation += [
                transforms.RandomCrop(random_crop_size, padding=4),
                transforms.RandomHorizontalFlip()
            ]
        train_transformation += transformation_list
        train_dataset_transform = transforms.Compose(train_transformation)
        train_dataset = dataset_map[config.dataset](
            "../data", train=True, download=True, transform=train_dataset_transform)
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size])
        val_dataset = copy.deepcopy(val_dataset)
        val_dataset.dataset = dataset_map[config.dataset](
            "../data", train=True, download=True, transform=dataset_transform)
        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True)
        val_dataset = BalancedDataLoader(
            train_dataset.dataset, selected_indices=val_dataset.indices)
        # as the validation is used for MIp and needs to be balanced
        val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
    else:
        raise Exception(
            'Dataset indx {} is not right'.format(str(config.dataset)))
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    parser = process_args()
    config = parser.parse_args()
    train_loader, val_loader, test_loader = prepare_dataset(config)
    X, _ = next(iter(val_loader))
    if config.model > 3:
        # conv model
        input_size = X.shape[-2:]
    else:
        input_size = X.flatten().cpu().numpy().reshape(
            X.shape[0], -1).shape[-1]
    n_channels = X.shape[1]
    n_output_classes = len(train_loader.dataset.dataset.class_to_idx.values())
    for exp_indx in range(config.num_resets):
        model_train = prepare_model_train(
            config, input_size, n_channels=n_channels, n_output_classes=n_output_classes, exp_indx=exp_indx)
        model_train.train(train_loader, val_loader=None,
                          num_epochs=config.epochs)
        model_train.print_results(train_loader, val_loader, test_loader)
        if config.retrain and config.other_model_dir is not None:
            model_train._logger.info('Finished Generalization from {} for Model {}'.format(
                config.other_model_dir, model_train.model.name))
            break
