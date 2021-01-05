import os
from training.utils import Mode, log_mean_std, log_config
from sparsify import SparsifyModel
from run_sparsify import prepare_arguments, prepare_images_mip_input
from dataset import MIPBatchLoader
from train_model import (
    model_indx_is_conv,
    prepare_model_train,
    prepare_dataset,
)
from visualization import plot_df, create_dataframe, plot_original_masked
from training.utils import save_pickle

"""script used to plot effect of changing values of lambdas
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
        prefix="_multi_lambdas",
    )
    log_config(model_train._logger, config)
    model_train.train(train_loader, val_loader=None, num_epochs=config.epochs)
    mip_data_loader.set_model(model_train.model)
    X, y, initial_bounds = next(mip_data_loader)

    parameters_removed_percentage_list = []
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

    n_lambdas = config.n_experiments
    x_data = []
    for lambda_value in range(1, n_lambdas):
        sparsify = SparsifyModel(
            model_train,
            threshold=config.threshold,
            sparsification_weight=lambda_value,
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
        x_data.append(lambda_value)

    data_points = {
        "masked_train_acc": masked_train_acc,
        "masked_train_loss": masked_train_loss,
        "masked_test_acc": masked_test_acc,
        "masked_test_loss": masked_test_loss,
        "original_model_results": original_model_results[0],
        "parameters_removed_percentage_list": parameters_removed_percentage_list,
    }
    save_pickle(
        os.path.join(model_train.storage_parent_dir, "different_lambdas_data.pickle"),
        data_points,
    )

    # Plot Train data
    original_model_train_acc = [
        original_model_results[0]["acc_train"]
        for _ in range(len(parameters_removed_percentage_list))
    ]
    plot_original_masked(
        x_data,
        original_model_train_acc,
        masked_train_acc,
        "Train Accuracy",
        "Lambda value",
        model_train.storage_parent_dir,
        disable_x_axis=False,
        step_size=4,
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
        "Lambda Value",
        model_train.storage_parent_dir,
        disable_x_axis=False,
        step_size=4,
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
        "Lambda Value",
        model_train.storage_parent_dir,
        disable_x_axis=False,
        step_size=4,
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
        "Lambda Value",
        model_train.storage_parent_dir,
        disable_x_axis=False,
        step_size=4,
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
        xlabel="Lambda Value",
        disable_x_axis=False,
        step_size=4,
    )
    log_mean_std(model_train._logger, "Original Train Acc", original_model_train_acc)
    log_mean_std(model_train._logger, "Masked Train Acc", masked_train_acc)
    log_mean_std(model_train._logger, "Original Test Acc", original_model_test_acc)
    log_mean_std(model_train._logger, "Masked Test Acc", masked_test_acc)
    log_mean_std(
        model_train._logger, "Pruning Percentage", parameters_removed_percentage_list
    )
