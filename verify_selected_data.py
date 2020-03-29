from training import ModelTrain, Logger
import os
import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from training import SparsifyModel, device
import numpy as np
from training.utils import Mode, get_storage_dir
from sparsify import prepare_config, prepare_images_mip_input
from train_model import learning_rates, dataset_map, model_indx_map, prepare_model_train, prepare_dataset
import math
import pandas as pd
from visualization import plot_df, create_dataframe
from training.utils import save_pickle, test_batch, device

"""script to verify robustness of introduced approach
"""

def plot_original_masked(x_data, original_result, masked_result, ylabel, xlabel, storage_parent_dir, disable_x_axis=True):
    dataframe_masked = create_dataframe(
        x_data, masked_result, 'Masked Model')
    dataframe_original = create_dataframe(
        x_data, original_result, 'Original Model')
    data_frame_all = pd.concat([dataframe_masked, dataframe_original])
    file_path = os.path.join(storage_parent_dir, '{}.jpg'.format(
        ylabel.strip().lower().replace(' ', '_')))
    plot_df(data_frame_all, file_path, ylabel=ylabel,
            xlabel=xlabel, disable_x_axis=disable_x_axis)


if __name__ == '__main__':
    config = prepare_config()
    train_loader, val_loader, test_loader = prepare_dataset(config)
    n_batches = len(val_loader)
    batch_size = val_loader.batch_size

    X, y, initial_bounds, input_size, n_output_classes, n_channels = prepare_images_mip_input(
        config, val_loader)
    model_train = prepare_model_train(config, input_size, n_channels=n_channels,
                                      n_output_classes=n_output_classes, exp_indx=0, prefix='_verify')

    model_train.train(train_loader, val_loader=None,
                      num_epochs=config.epochs)
    model_train._logger.info(
        'Started Model {} verification by changing input to MIP'.format(model_train.model.name))
    n_subselect = int(math.ceil(batch_size / config.num_samples))
    # testing original model results
    # its results should be a flat one
    original_model_results = model_train.print_results(
        train_loader, None, test_loader, test_original_model=True, test_masked_model=False)
    masked_train_acc = []
    masked_train_loss = []
    masked_test_acc = []
    masked_test_loss = []
    val_batch_accuracy = []
    val_batch_loss = []
    parameters_removed_percentage_list = []
    n_experiments = 25
    for _ in range(n_experiments):
        batch_indx = np.random.randint(0, n_batches)
        subset_indx = np.random.randint(0, n_subselect)
        X, y, initial_bounds, input_size, _, _ = prepare_images_mip_input(
            config, val_loader, start_indx=subset_indx * config.num_samples, batch_indx=batch_indx)
        n_correct_itms, loss_value = test_batch(
            model_train.model, X, torch.from_numpy(y).to(device))
        loss_value /= len(y)
        batch_accuracy = n_correct_itms * 100 / len(y)
        sparsify = SparsifyModel(model_train, threshold=config.threshold,
                                 sparsification_weight=config.sparsification_weight)
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
        val_batch_accuracy.append(batch_accuracy)
        val_batch_loss.append(loss_value)

    data_points = {
        'masked_train_acc': masked_train_acc,
        'masked_train_loss': masked_train_loss,
        'masked_test_acc': masked_test_acc,
        'masked_test_loss': masked_test_loss,
        'original_model_results': original_model_results[0],
        'parameters_removed_percentage_list': parameters_removed_percentage_list,
        'val_batch_accuracy': val_batch_accuracy,
        'val_batch_loss': val_batch_loss
    }
    save_pickle(os.path.join(model_train.storage_parent_dir,
                             'plotting_data_points.pickle'), data_points)

    # Plot Train data
    x_data = [indx for indx in range(len(parameters_removed_percentage_list))]
    original_model_train_acc = [original_model_results[0]['acc_train']
                                for _ in range(len(parameters_removed_percentage_list))]
    plot_original_masked(x_data, original_model_train_acc, masked_train_acc,
                         'Train Accuracy', 'batch index', model_train.storage_parent_dir)

    # Plot Test Data
    original_model_test_acc = [original_model_results[0]['acc_test']
                               for _ in range(len(parameters_removed_percentage_list))]
    plot_original_masked(x_data, original_model_test_acc, masked_test_acc,
                         'Test Accuracy', 'batch index', model_train.storage_parent_dir)

    # Plot loss Train data
    original_model_train_loss = [original_model_results[0]['loss_train']
                                 for _ in range(len(parameters_removed_percentage_list))]
    plot_original_masked(x_data, original_model_train_loss, masked_train_loss,
                         'Train Loss', 'batch index', model_train.storage_parent_dir)

    # Plot Test Data
    original_model_test_loss = [original_model_results[0]['loss_test']
                                for _ in range(len(parameters_removed_percentage_list))]
    plot_original_masked(x_data, original_model_test_loss, masked_test_loss,
                         'Test Loss', 'batch index', model_train.storage_parent_dir)

    # Plot Percentage of removal on different input data
    dataframe_masked = create_dataframe(
        x_data, parameters_removed_percentage_list, '')
    pruning_file_path = os.path.join(
        model_train.storage_parent_dir, 'pruning_percentage.jpg')
    plot_df(dataframe_masked, pruning_file_path, ylabel='Pruning Percentage',
            xlabel='batch index', disable_x_axis=True)

    # Plot Accuracy of validation batch on input data
    dataframe_masked = create_dataframe(
        x_data, val_batch_accuracy, '')
    batch_accuracy_path = os.path.join(
        model_train.storage_parent_dir, 'val_batch_accuracy.jpg')
    plot_df(dataframe_masked, batch_accuracy_path, ylabel='Batch Accuracy',
            xlabel='batch index', disable_x_axis=True)

    # Plot Loss of validation batch on input data
    dataframe_masked = create_dataframe(
        x_data, val_batch_loss, '')
    batch_loss_path = os.path.join(
        model_train.storage_parent_dir, 'val_batch_loss.jpg')
    plot_df(dataframe_masked, batch_loss_path, ylabel='Batch Loss',
            xlabel='batch index', disable_x_axis=True)

    def log_info(suffix, data):
        model_train._logger.info(
            '{} mean {} +-/ {}'.format(suffix, np.mean(data), np.std(data)))

    log_info('Original Train Acc', original_model_train_acc)
    log_info('Masked Train Acc', masked_train_acc)
    log_info('Original Test Acc', original_model_test_acc)
    log_info('Masked Test Acc', masked_test_acc)
    log_info('Pruning Percentage', parameters_removed_percentage_list)
    log_info('Batch Accuracy Original Model', val_batch_accuracy)
    log_info('Batch Loss Original Model', val_batch_loss)
