import torch
from training import device
from dataset import MIPBatchLoader
from run_sparsify import prepare_images_mip_input, prepare_arguments
from train_model import (
    model_indx_is_conv,
    prepare_model_train,
    prepare_dataset,
)
from related_pruning import SNIP
from torch import nn as nn
import time
import copy

"""A script calling SNIP to prune the model
https://arxiv.org/abs/1810.02340
"""


def prepare_config():
    parser = prepare_arguments()
    parser.add_argument("--keep", "-kp", default=0.45, type=float)
    config = parser.parse_args()
    return config


def apply_prune_mask(net, keep_masks):

    # Before I can zip() layers and pruning masks I need to make sure they match
    # one-to-one by removing all the irrelevant modules:
    prunable_layers = filter(
        lambda layer: isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear),
        net.modules(),
    )

    for layer, keep_mask in zip(prunable_layers, keep_masks):
        assert layer.weight.shape == keep_mask.shape

        def hook_factory(keep_mask):
            """
            The hook function can't be defined directly here because of Python's
            late binding which would result in all hooks getting the very last
            mask! Getting it through another function forces early binding.
            """

            def hook(grads):
                return grads * keep_mask

            return hook

        # mask[i] == 0 --> Prune parameter
        # mask[i] == 1 --> Keep parameter

        # Step 1: Set the masked weights to zero (NB the biases are ignored)
        # Step 2: Make sure their gradients remain zero
        layer.weight.data[keep_mask == 0.0] = 0.0
        layer.weight.register_hook(hook_factory(keep_mask))


if __name__ == "__main__":
    config = prepare_config()
    data_loaders = prepare_dataset(config)
    train_loader = data_loaders['train']
    val_loader = data_loaders['val']
    test_loader = data_loaders['test']
    n_batches = len(val_loader)
    batch_size = val_loader.batch_size
    mip_data_loader = MIPBatchLoader(
        config, val_loader, epsilon=1e-5, is_conv_model=model_indx_is_conv[config.model]
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
        exp_indx=0,
        prefix="_snip",
    )
    start_time = time.time()
    keep_masks = SNIP(model_train.model, config.keep, train_loader, device)
    n_pruned_params = torch.sum(torch.cat([torch.flatten(x == 0) for x in keep_masks]))
    n_kept_params = torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks]))
    percentage_pruning = n_pruned_params * 100.0 / (n_pruned_params + n_kept_params)
    masked_model = copy.deepcopy(model_train.model)
    apply_prune_mask(masked_model, keep_masks)
    model_train._logger.info(
        "[Exp] Snip compression took {} seconds to compress {} with {} %".format(
            str(time.time() - start_time), model_train.model.name, percentage_pruning
        )
    )

    model_train.train(train_loader, val_loader=None, num_epochs=config.epochs)

    model_train._logger.info("Original Model accuracy")
    model_results = model_train.print_results(
        train_loader,
        val_loader,
        test_loader,
        test_original_model=True,
        test_masked_model=False,
    )
    model_train._logger.info("Masked Model accuracy")
    model_train.model_masked = masked_model
    model_train.train(
        train_loader, val_loader=None, num_epochs=config.epochs, finetune_masked=True
    )
    model_results = model_train.print_results(
        train_loader,
        val_loader,
        test_loader,
        test_original_model=False,
        test_masked_model=True,
    )
