import torch
from training import device, Mode
from training.utils import log_mean_std
from dataset import MIPBatchLoader
from run_sparsify import prepare_images_mip_input, prepare_arguments
from train_model import (
    model_indx_is_conv,
    prepare_model_train,
    prepare_dataset,
    log_config,
)
from related_pruning import SNIP
from torch import nn as nn
import time
import numpy as np
import copy

"""A script calling SNIP to prune the model
https://arxiv.org/abs/1810.02340
"""


def prepare_config():
    parser = prepare_arguments()
    parser.add_argument("--keep", "-kp", default=0.45, type=float)
    parser.add_argument("--mip-batch", "-mbatch", action="store_true")
    parser.add_argument("--prune-neurons", "-prunen", action="store_true")
    # add pruning after training
    parser.add_argument("--trained-net", "-trainedn", action="store_true")
    config = parser.parse_args()
    return config


def apply_snip(
    model_train, keep_value, inputs, targets, prune_neurons=False, trained_network=False
):
    start_time = time.time()
    keep_masks = SNIP(
        model_train.model,
        keep_value,
        inputs,
        targets,
        prune_neurons=prune_neurons,
        trained_network=trained_network,
    )
    n_pruned_params = torch.sum(torch.cat([torch.flatten(x == 0) for x in keep_masks]))
    n_kept_params = torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks]))
    percentage_pruning = n_pruned_params * 100.0 / (n_pruned_params + n_kept_params)
    computation_time = time.time() - start_time
    masking_indices = {}
    indx_keep_mask = 0
    for layer_indx, layer in enumerate(model_train.model):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            if prune_neurons:
                masking_indices[layer_indx] = np.where(
                    keep_masks[indx_keep_mask].cpu().numpy() == 0
                )[0]
            else:
                masking_indices[layer_indx] = (keep_masks[indx_keep_mask] == 0).cpu()
            indx_keep_mask += 1
    return masking_indices, percentage_pruning.item(), computation_time


if __name__ == "__main__":
    config = prepare_config()
    data_loaders = prepare_dataset(config)
    train_loader = data_loaders["train"]
    val_loader = data_loaders["val"]
    test_loader = data_loaders["test"]
    n_batches = len(val_loader)
    batch_size = val_loader.batch_size
    n_experiments = config.num_resets
    parameters_removed_percentage_list = []
    original_train_acc = []
    original_test_acc = []
    masked_train_acc = []
    masked_test_acc = []
    computation_time_list = []

    small_masked_train_acc = []
    small_masked_test_acc = []
    parameters_removed_small_batch = []
    model_train = None
    for i in range(n_experiments):
        mip_data_loader = MIPBatchLoader(
            config,
            val_loader,
            epsilon=1e-5,
            is_conv_model=model_indx_is_conv[config.model],
        )
        (
            X,
            y,
            initial_bounds,
            input_size,
            n_output_classes,
            n_channels,
        ) = prepare_images_mip_input(mip_data_loader)
        model_train = prepare_model_train(
            config,
            input_size,
            n_channels=n_channels,
            n_output_classes=n_output_classes,
            exp_indx=i,
            prefix="_snip",
        )
        if i == 0:
            log_config(model_train._logger, config)
        model_train._logger.info(
            "[Exp] Started SNIP compression experiment #{}".format(i)
        )
        inputs, targets = next(iter(train_loader))
        inputs = inputs.to(device)
        targets = targets.to(device)
        # backing the model's initialization
        model_initialization = copy.deepcopy(model_train.get_model_to_sparsify())
        model_train.train(train_loader, val_loader=None, num_epochs=config.epochs)

        model_train._logger.info("Original Model accuracy")
        model_results = model_train.print_results(
            train_loader,
            val_loader,
            test_loader,
            test_original_model=True,
            test_masked_model=False,
        )
        original_train_acc.append(model_results[0]["acc_train"])
        original_test_acc.append(model_results[0]["acc_test"])
        model_train._logger.info("Masked Model accuracy")
        if not (config.trained_net):
            model_train.set_model_to_sparsify(model_initialization)
        masking_indices, percentage_pruning, computation_time = apply_snip(
            model_train,
            config.keep,
            inputs,
            targets,
            prune_neurons=config.prune_neurons,
            trained_network=config.trained_net,
        )
        parameters_removed_percentage_list.append(percentage_pruning)
        model_train._logger.info(
            "[Exp] Snip compression took {} seconds to compress {} with {} %".format(
                str(computation_time), model_train.model.name, percentage_pruning,
            )
        )
        computation_time_list.append(computation_time)
        model_train.set_mask_indices(
            masking_indices, suffix="_" + Mode.MASK.name, save_masking_model=True
        )
        if not (config.trained_net):
            model_train.train(
                train_loader,
                val_loader=None,
                num_epochs=config.epochs,
                finetune_masked=True,
            )
        model_results = model_train.print_results(
            train_loader,
            val_loader,
            test_loader,
            test_original_model=False,
            test_masked_model=True,
        )
        masked_train_acc.append(model_results[0]["acc_train"])
        masked_test_acc.append(model_results[0]["acc_test"])

        if config.mip_batch:
            start_time = time.time()
            if not (config.trained_net):
                model_train.set_model_to_sparsify(model_initialization)
            masking_indices, percentage_pruning, computation_time = apply_snip(
                model_train,
                config.keep,
                inputs,
                targets,
                prune_neurons=config.prune_neurons,
                trained_network=config.trained_net,
            )
            parameters_removed_small_batch.append(percentage_pruning)
            model_train._logger.info(
                "[Exp] Snip compression with MIP batch took {} seconds to compress {} with {} %".format(
                    str(computation_time), model_train.model.name, percentage_pruning,
                )
            )
            computation_time_list.append(computation_time)

            model_train._logger.info("MIP Batch Masked Model accuracy")
            model_train.set_mask_indices(
                masking_indices,
                suffix="_" + Mode.MASK.name + "_MIP",
                save_masking_model=False,
            )
            if not (config.trained_net):
                model_train.train(
                    train_loader,
                    val_loader=None,
                    num_epochs=config.epochs,
                    finetune_masked=True,
                )
            model_results = model_train.print_results(
                train_loader,
                val_loader,
                test_loader,
                test_original_model=False,
                test_masked_model=True,
            )
            small_masked_train_acc.append(model_results[0]["acc_train"])
            small_masked_test_acc.append(model_results[0]["acc_test"])

    log_mean_std(
        model_train._logger, "Original Train Acc", original_train_acc,
    )
    log_mean_std(
        model_train._logger, "Original Test Acc", original_test_acc,
    )
    log_mean_std(model_train._logger, "Masked Train Acc", masked_train_acc)
    log_mean_std(model_train._logger, "Masked Test Acc", masked_test_acc)
    if config.mip_batch:
        log_mean_std(
            model_train._logger, "MIP Batch Masked Train Acc", small_masked_train_acc
        )
        log_mean_std(
            model_train._logger, "MIP Batch  Masked Test Acc", small_masked_test_acc
        )
        log_mean_std(
            model_train._logger,
            "Pruning Percentage MIP Batch",
            parameters_removed_small_batch,
        )

    log_mean_std(
        model_train._logger, "Pruning Percentage", parameters_removed_percentage_list
    )
    log_mean_std(model_train._logger, "Computation Time", computation_time_list)
