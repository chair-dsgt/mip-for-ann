from training import ModelTrain, Logger
import os
import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from train_model import learning_rates, dataset_map, model_indx_map, process_args, prepare_model_train, prepare_dataset, prepare_model
from training import SparsifyModel, device
import numpy as np
from training.utils import Mode, get_storage_dir, test_batch
import time

"""script used to sparsify a trained model by computing the importance score of each neuron and pruning non-critical neurons
"""


def prepare_arguments():
    parser = process_args()
    parser.add_argument('--num-samples', '-n', default=10, type=int)
    parser.add_argument('--sparsification-weight', '-sw', default=5, type=int)
    parser.add_argument('--threshold', '-tt', default=1e-1, type=float)
    parser.add_argument('--force', '-f', action='store_true')
    parser.add_argument('--heat-map', '-hm', action='store_true')
    parser.add_argument('--fine-tune', '-ft', action='store_true')
    parser.add_argument('--relaxed', '-rl', action='store_true')
    parser.add_argument('--tuned-epochs', '-te', default=1, type=int)
    return parser


def prepare_config():
    parser = prepare_arguments()
    config = parser.parse_args()
    return config


def get_mip_images(config, data_loader, epsilon, start_indx, batch_indx, model=None):
    check_mip_input_acc = True
    if model is None:
        check_mip_input_acc = False
    end_indx = start_indx + config.num_samples
    if end_indx > data_loader.batch_size:
        end_indx = data_loader.batch_size - 1
    for data_indx, data_itm in enumerate(data_loader):
        X, y = data_itm
        # loading a single batch
        if check_mip_input_acc:
            X, y = X.to(device), y.to(device)
            n_correct, _ = test_batch(
                model, X[start_indx:end_indx], y[start_indx:end_indx])
            if n_correct / len(y) > 0.95:
                break
        else:
            if data_indx == batch_indx:
                X, y = X.to(device), y.to(device)
                break

    X = X[start_indx:end_indx]
    y = y[start_indx:end_indx]
    initial_bounds = (X.clamp(min=0) - epsilon, X.clamp(max=1) + epsilon)
    y = y.cpu().numpy()
    return X, y, initial_bounds


def prepare_images_mip_input(config, data_loader, start_indx=0, batch_indx=0, epsilon=1e-5):
    X, y, initial_bounds = get_mip_images(
        config, data_loader, epsilon, start_indx, batch_indx)
    n_channels = X.shape[1]
    if config.model > 3:
        # conv model
        input_size = X.shape[-2:]
    else:
        input_size = X.flatten().cpu().numpy().reshape(
            X.shape[0], -1).shape[-1]
    n_output_classes = len(data_loader.dataset.dataset.class_to_idx.values())
    return X, y, initial_bounds, input_size, n_output_classes, n_channels


if __name__ == '__main__':
    config = prepare_config()
    train_loader, val_loader, test_loader = prepare_dataset(config)
    X, y, initial_bounds, input_size, n_output_classes, n_channels = prepare_images_mip_input(
        config, val_loader)
    exp_indx = 0
    all_exp_results = {}
    finetuning_prefix = 'Fine Tuned'
    pruning_percentage_prefix = 'Pruning Percentage'
    tuning_epochs = config.tuned_epochs

    sparsification_time_list = []
    while True:
        model = prepare_model(config, input_size=input_size,
                              n_channels=n_channels, n_output_classes=n_output_classes)
        storage_parent_dir = config.storage_dir
        storage_parent_dir = get_storage_dir(
            storage_parent_dir, config, model.name, exp_indx)
        del model
        if not(os.path.isdir(storage_parent_dir)):
            break
        model_train = prepare_model_train(
            config, input_size, n_channels=n_channels, n_output_classes=n_output_classes, exp_indx=exp_indx)

        X, y, initial_bounds = get_mip_images(
            config, val_loader, epsilon=1e-5, start_indx=0, batch_indx=0, model=model_train.model)

        sparsify = SparsifyModel(model_train, threshold=config.threshold,
                                 sparsification_weight=config.sparsification_weight, relaxed_constraints=config.relaxed)
        sparsify.create_bounds(initial_bounds)
        parameters_removed_percentage = 0
        for mode in [Mode.MASK, Mode.Random, Mode.CRITICAL]:
            start_sparsify_time = time.time()
            removal_percentage = sparsify.sparsify_model(
                X, y, mode=mode, use_cached=not(config.force))
            sparsification_time = time.time() - start_sparsify_time
            if mode == Mode.MASK:
                parameters_removed_percentage = removal_percentage
                sparsification_time_list.append(sparsification_time)
            model_results = model_train.print_results(
                train_loader, val_loader, test_loader, save_heat_map=config.heat_map, mode_name=mode.name)

            model_results[-1][pruning_percentage_prefix] = parameters_removed_percentage
            for model_indx, mode_name in enumerate(['original', mode.name]):
                if (pruning_percentage_prefix) not in model_results[model_indx]:
                    model_results[model_indx][pruning_percentage_prefix] = 0
                for metric_name in model_results[model_indx]:
                    key_results_name = mode_name + metric_name
                    if key_results_name not in all_exp_results:
                        all_exp_results[key_results_name] = []
                    all_exp_results[key_results_name].append(
                        model_results[model_indx][metric_name])
            if mode == Mode.MASK and config.fine_tune and tuning_epochs > 0:
                model_train.train(train_loader, val_loader=None,
                                  num_epochs=tuning_epochs, finetune_masked=True)
                finetuned_model_results = model_train.print_results(
                    train_loader, val_loader, test_loader, mode_name=mode.name, test_original_model=False)
                finetuned_model_results[0][pruning_percentage_prefix] = parameters_removed_percentage
                for metric_name in finetuned_model_results[0]:
                    key_results_name = finetuning_prefix + metric_name
                    if key_results_name not in all_exp_results:
                        all_exp_results[key_results_name] = []
                    all_exp_results[key_results_name].append(
                        finetuned_model_results[0][metric_name])
        exp_indx += 1
    # Now logging mean and variance of sparsification results
    if exp_indx > 0:
        list_modes = ['original', Mode.MASK.name, Mode.Random.name,
                      Mode.CRITICAL.name]
        if config.fine_tune:
            list_modes.insert(2, finetuning_prefix)
        for mode in list_modes:
            for metric_name in ['loss_train', 'acc_train', 'loss_test', 'acc_test', pruning_percentage_prefix]:
                results_list = all_exp_results[mode + metric_name]
                metric_name_clean = ' '.join(
                    metric_name.split('_')).capitalize()
                model_train._logger.info('{} {} mean {} +- {}'.format(
                    mode, metric_name_clean, np.mean(results_list), np.std(results_list)))

        model_train._logger.info('Sparsification time including swapping pytorch layers mean {} +- {}'.format(
            np.mean(sparsification_time_list), np.std(sparsification_time_list)))
