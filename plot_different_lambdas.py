from training import ModelTrain, Logger
import os
import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from training import SparsifyModel, device
import numpy as np
from training.utils import Mode, get_storage_dir
from sparsify import prepare_config, prepare_images_mip_input, get_mip_images
from train_model import learning_rates, dataset_map, model_indx_map, prepare_model_train, prepare_dataset
import math
import pandas as pd
from visualization import plot_df, create_dataframe
from training.utils import save_pickle
from verify_selected_data import plot_original_masked

"""script used to plot effect of changing values of lambdas
"""
if __name__ == '__main__':
    config = prepare_config()
    train_loader, val_loader, test_loader = prepare_dataset(config)

    X, y, initial_bounds, input_size, n_output_classes, n_channels = prepare_images_mip_input(
        config, train_loader, start_indx=0, batch_indx=2)
    model_train = prepare_model_train(config, input_size, n_channels=n_channels,
                                      n_output_classes=n_output_classes, exp_indx=0, prefix='_multi_lambdas')

    model_train.train(train_loader, val_loader=None,
                      num_epochs=config.epochs)

    X, y, initial_bounds = get_mip_images(
        config, train_loader, epsilon=1e-5, start_indx=0, batch_indx=0, model=model_train.model)

    parameters_removed_percentage_list = []
    original_model_results = model_train.print_results(
        train_loader, None, test_loader, test_original_model=True, test_masked_model=False)
    masked_train_acc = []
    masked_train_loss = []
    masked_test_acc = []
    masked_test_loss = []

    n_lambdas = 25  
    x_data = []
    for lambda_value in range(1, n_lambdas):
        sparsify = SparsifyModel(model_train, threshold=config.threshold,
                                 sparsification_weight=lambda_value)
        sparsify.create_bounds(initial_bounds)

        parameters_removed_percentage = sparsify.sparsify_model(
            X, y, mode=Mode.MASK, use_cached=False)
        parameters_removed_percentage_list.append(
            parameters_removed_percentage)
        masked_model_results = model_train.print_results(
            train_loader, None, test_loader, test_original_model=False, test_masked_model=True)
        masked_train_acc.append(masked_model_results[0]['acc_train'])
        masked_train_loss.append(masked_model_results[0]['loss_train'])
        masked_test_acc.append(masked_model_results[0]['acc_test'])
        masked_test_loss.append(masked_model_results[0]['loss_test'])
        x_data.append(lambda_value)

    data_points = {
        'masked_train_acc': masked_train_acc,
        'masked_train_loss': masked_train_loss,
        'masked_test_acc': masked_test_acc,
        'masked_test_loss': masked_test_loss,
        'original_model_results': original_model_results[0],
        'parameters_removed_percentage_list': parameters_removed_percentage_list
    }
    save_pickle(os.path.join(model_train.storage_parent_dir,
                             'different_lambdas_data.pickle'), data_points)

    # Plot Train data
    original_model_train_acc = [original_model_results[0]['acc_train']
                                for _ in range(len(parameters_removed_percentage_list))]
    plot_original_masked(x_data, original_model_train_acc, masked_train_acc,
                         'Train Accuracy', 'Lambda value', model_train.storage_parent_dir, disable_x_axis=False)

    # Plot Test Data
    original_model_test_acc = [original_model_results[0]['acc_test']
                               for _ in range(len(parameters_removed_percentage_list))]
    plot_original_masked(x_data, original_model_test_acc, masked_test_acc,
                         'Test Accuracy', 'Lambda Value', model_train.storage_parent_dir, disable_x_axis=False)

    # Plot loss Train data
    original_model_train_loss = [original_model_results[0]['loss_train']
                                 for _ in range(len(parameters_removed_percentage_list))]
    plot_original_masked(x_data, original_model_train_loss, masked_train_loss,
                         'Train Loss', 'Lambda Value', model_train.storage_parent_dir, disable_x_axis=False)

    # Plot Test Data
    original_model_test_loss = [original_model_results[0]['loss_test']
                                for _ in range(len(parameters_removed_percentage_list))]
    plot_original_masked(x_data, original_model_test_loss, masked_test_loss,
                         'Test Loss', 'Lambda Value', model_train.storage_parent_dir, disable_x_axis=False)

    # Plot Percentage of removal on different input data
    dataframe_masked = create_dataframe(
        x_data, parameters_removed_percentage_list, '')
    pruning_file_path = os.path.join(
        model_train.storage_parent_dir, 'pruning_percentage.jpg')
    plot_df(dataframe_masked, pruning_file_path, ylabel='Pruning Percentage',
            xlabel='Lambda Value', disable_x_axis=False)

    def log_info(suffix, data):
        model_train._logger.info(
            '{} mean {} +-/ {}'.format(suffix, np.mean(data), np.std(data)))

    log_info('Original Train Acc', original_model_train_acc)
    log_info('Masked Train Acc', masked_train_acc)
    log_info('Original Test Acc', original_model_test_acc)
    log_info('Masked Test Acc', masked_test_acc)
    log_info('Pruning Percentage', parameters_removed_percentage_list)
