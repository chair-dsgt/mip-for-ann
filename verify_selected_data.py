import os
import torch
from training import device
from sparsify import SparsifyModel
import numpy as np
from training.utils import Mode
from run_sparsify import prepare_arguments, prepare_images_mip_input
from train_model import (
    model_indx_is_conv,
    prepare_model_train,
    prepare_dataset,
)
import math
from visualization import plot_df, create_dataframe, plot_original_masked
from training.utils import save_pickle, test_batch, log_config
from dataset import MIPBatchLoader

"""script to verify robustness of introduced approach
"""


def prepare_config():
    parser = prepare_arguments()
    parser.add_argument("--n-experiments", "-nex", default=25, type=int)
    config = parser.parse_args()
    return config


if __name__ == "__main__":
    config = prepare_config()
    data_loaders = prepare_dataset(config)
    train_loader = data_loaders["train"]
    val_loader = data_loaders["val"]
    test_loader = data_loaders["test"]
    n_batches = len(val_loader)
    batch_size = val_loader.batch_size
    mip_data_loader = MIPBatchLoader(
        config,
        val_loader,
        epsilon=1e-5,
        random_order=True,
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
        exp_indx=0,
        prefix="_verify",
    )
    log_config(model_train._logger, config)
    model_train.train(train_loader, val_loader=None, num_epochs=config.epochs)
    model_train._logger.info(
        "Started Model {} verification by changing input to MIP".format(
            model_train.model.name
        )
    )
    n_subselect = int(math.ceil(batch_size / config.num_samples))
    # testing original model results
    # its results should be a flat one
    original_model_results = model_train.print_results(
        train_loader,
        None,
        test_loader,
        test_original_model=True,
        test_masked_model=False,
    )
    masked_train_acc = []
    masked_train_loss = []
    masked_test_acc = []
    masked_test_loss = []
    val_batch_accuracy = []
    val_batch_loss = []
    parameters_removed_percentage_list = []
    n_experiments = config.n_experiments
    for _ in range(n_experiments):
        X, y, initial_bounds = next(mip_data_loader)
        prediction_results = test_batch(
            model_train.model, X, torch.from_numpy(y).to(device)
        )
        loss_value = prediction_results.get_data_loss() / len(y)
        batch_accuracy = prediction_results.get_accuracy() * 100
        sparsify = SparsifyModel(
            model_train,
            threshold=config.threshold,
            sparsification_weight=config.sparsification_weight,
            relaxed_constraints=config.relaxed,
        )
        sparsify.create_bounds(initial_bounds)

        parameters_removed_percentage = sparsify.sparsify_model(
            X, y, mode=Mode.MASK, use_cached=False
        )
        parameters_removed_percentage_list.append(parameters_removed_percentage)

        masked_model_results = model_train.print_results(
            train_loader,
            None,
            test_loader,
            test_original_model=False,
            test_masked_model=True,
        )
        masked_train_acc.append(masked_model_results[0]["acc_train"])
        masked_train_loss.append(masked_model_results[0]["loss_train"])
        masked_test_acc.append(masked_model_results[0]["acc_test"])
        masked_test_loss.append(masked_model_results[0]["loss_test"])
        val_batch_accuracy.append(batch_accuracy)
        val_batch_loss.append(loss_value)

    data_points = {
        "masked_train_acc": masked_train_acc,
        "masked_train_loss": masked_train_loss,
        "masked_test_acc": masked_test_acc,
        "masked_test_loss": masked_test_loss,
        "original_model_results": original_model_results[0],
        "parameters_removed_percentage_list": parameters_removed_percentage_list,
        "val_batch_accuracy": val_batch_accuracy,
        "val_batch_loss": val_batch_loss,
    }
    save_pickle(
        os.path.join(model_train.storage_parent_dir, "plotting_data_points.pickle"),
        data_points,
    )

    # Plot Train data
    x_data = [indx for indx in range(len(parameters_removed_percentage_list))]
    original_model_train_acc = [
        original_model_results[0]["acc_train"]
        for _ in range(len(parameters_removed_percentage_list))
    ]
    plot_original_masked(
        x_data,
        original_model_train_acc,
        masked_train_acc,
        "Train Accuracy",
        "batch index",
        model_train.storage_parent_dir,
        disable_x_axis=False,
        step_size=5,
    )

    # Plot Test Data
    original_model_test_acc = [
        original_model_results[0]["acc_test"]
        for _ in range(len(parameters_removed_percentage_list))
    ]
    plot_original_masked(
        x_data,
        original_model_test_acc,
        masked_test_acc,
        "Test Accuracy",
        "batch index",
        model_train.storage_parent_dir,
        disable_x_axis=False,
        step_size=5,
    )

    # Plot loss Train data
    original_model_train_loss = [
        original_model_results[0]["loss_train"]
        for _ in range(len(parameters_removed_percentage_list))
    ]
    plot_original_masked(
        x_data,
        original_model_train_loss,
        masked_train_loss,
        "Train Loss",
        "batch index",
        model_train.storage_parent_dir,
        disable_x_axis=False,
        step_size=5,
    )

    # Plot Test Data
    original_model_test_loss = [
        original_model_results[0]["loss_test"]
        for _ in range(len(parameters_removed_percentage_list))
    ]
    plot_original_masked(
        x_data,
        original_model_test_loss,
        masked_test_loss,
        "Test Loss",
        "batch index",
        model_train.storage_parent_dir,
        disable_x_axis=False,
        step_size=5,
    )

    # Plot Percentage of removal on different input data
    dataframe_masked = create_dataframe(x_data, parameters_removed_percentage_list, "")
    pruning_file_path = os.path.join(
        model_train.storage_parent_dir, "pruning_percentage.jpg"
    )
    plot_df(
        dataframe_masked,
        pruning_file_path,
        ylabel="Pruning Percentage",
        xlabel="batch index",
        disable_x_axis=False,
        step_size=5,
    )

    # Plot Accuracy of validation batch on input data
    dataframe_masked = create_dataframe(x_data, val_batch_accuracy, "")
    batch_accuracy_path = os.path.join(
        model_train.storage_parent_dir, "val_batch_accuracy.jpg"
    )
    plot_df(
        dataframe_masked,
        batch_accuracy_path,
        ylabel="Batch Accuracy",
        xlabel="batch index",
        disable_x_axis=False,
        step_size=5,
    )

    # Plot Accuracy of validation batch on input data
    dataframe_masked = create_dataframe(x_data, val_batch_loss, "")
    batch_loss_path = os.path.join(model_train.storage_parent_dir, "val_batch_loss.jpg")
    plot_df(
        dataframe_masked,
        batch_loss_path,
        ylabel="Batch Loss",
        xlabel="batch index",
        disable_x_axis=False,
        step_size=5,
    )

    def log_info(suffix, data):
        model_train._logger.info(
            "{} mean {} +-/ {}".format(suffix, np.mean(data), np.std(data))
        )

    log_info("Original Train Acc", original_model_train_acc)
    log_info("Masked Train Acc", masked_train_acc)
    log_info("Original Test Acc", original_model_test_acc)
    log_info("Masked Test Acc", masked_test_acc)
    log_info("Pruning Percentage", parameters_removed_percentage_list)
    log_info("Batch Accuracy Original Model", val_batch_accuracy)
    log_info("Batch Loss Original Model", val_batch_loss)
