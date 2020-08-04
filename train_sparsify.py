import os
from training.utils import Mode
from sparsify import SparsifyModel
from run_sparsify import prepare_arguments, prepare_images_mip_input
from train_model import (
    model_indx_is_conv,
    prepare_model_train,
    prepare_dataset,
)

from visualization import plot_df, create_dataframe, plot_original_masked
from training.utils import save_pickle, log_config, log_mean_std
from dataset import MIPBatchLoader

"""script to sparsify model every epoch or every n iterations
"""


def prepare_config():
    parser = prepare_arguments()
    # to run sparsification every n steps
    parser.add_argument("--step", "-trst", action="store_true")
    # run sparsification every n train step / epoch
    parser.add_argument("--every-n", "-ent", default=1, type=int)
    # run incremental sparsification 
    parser.add_argument("--incremental", "-incr", action="store_true")
    config = parser.parse_args()
    return config


def plot_pruning_evolution(out_file_path, x_data, y_data, ylabel, xlabel="Iteration"):
    data_frame = create_dataframe(x_data, y_data, "")
    plot_df(
        data_frame, out_file_path, ylabel=ylabel, xlabel=xlabel, disable_x_axis=True,
    )


if __name__ == "__main__":
    config = prepare_config()
    data_loaders = prepare_dataset(config)
    train_loader = data_loaders['train']
    val_loader = data_loaders['val']
    test_loader = data_loaders['test']
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
        prefix="_train_sparsify",
        use_cached=False,
        incremental_sparsify=config.incremental
    )
    log_config(model_train._logger, config)
    x_label_data_list = []
    parameter_removed_list = []
    model_results_list = []

    def sparsify_every(args):
        # args include epoch and step number
        train_step = 0
        train_epoch = args["epoch"]
        if config.step:
            train_step = args["step"]
            if train_step % config.every_n != 0:
                return
        elif train_epoch % config.every_n != 0:
            return

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
        model_results = model_train.print_results(
            train_loader,
            None,
            test_loader,
            test_original_model=True,
            test_masked_model=True,
        )
        model_train.model.train()
        model_train.model_masked.train()
        for param in model_train.model_masked.parameters():
            param.requires_grad = True
        model_train.model_masked.register_backward_hooks()
        model_results_list.append(model_results)
        parameter_removed_list.append(parameters_removed_percentage)
        if config.step:
            x_label_data_list.append(train_epoch * len(train_loader) + train_step)
        else:
            x_label_data_list.append(train_epoch)

    handler_name = "step" if config.step else "epoch"
    model_train.add_train_listener(handler_name, sparsify_every)
    model_train.train(train_loader, val_loader=None, num_epochs=config.epochs)

    masked_train_acc = []
    masked_train_loss = []
    masked_test_acc = []
    masked_test_loss = []
    original_train_acc = []
    original_train_loss = []
    original_test_acc = []
    original_test_loss = []
    for model_result in model_results_list:
        original_train_acc.append(model_result[0]["acc_train"])
        original_train_loss.append(model_result[0]["loss_train"])

        masked_train_acc.append(model_result[1]["acc_train"])
        masked_train_loss.append(model_result[1]["loss_train"])

        original_test_acc.append(model_result[0]["acc_test"])
        original_test_loss.append(model_result[0]["loss_test"])

        masked_test_acc.append(model_result[1]["acc_test"])
        masked_test_loss.append(model_result[1]["loss_test"])

    data_points = {
        "masked_train_acc": masked_train_acc,
        "masked_train_loss": masked_train_loss,
        "masked_test_acc": masked_test_acc,
        "masked_test_loss": masked_test_loss,
        "original_train_acc": original_train_acc,
        "original_test_acc": original_test_acc,
        "original_train_loss": original_train_loss,
        "original_test_loss": original_test_loss,
        "parameters_removed_percentage_list": parameter_removed_list,
        "iteration_number_list": x_label_data_list,
    }
    save_pickle(
        os.path.join(model_train.storage_parent_dir, "plotting_data_points.pickle"),
        data_points,
    )
    log_mean_std(model_train._logger, "Original Train Acc", original_train_acc)
    log_mean_std(model_train._logger, "Masked Train Acc", masked_train_acc)

    log_mean_std(model_train._logger, "Original Test Acc", original_test_acc)
    log_mean_std(model_train._logger, "Masked Test Acc", masked_test_acc)

    log_mean_std(model_train._logger, "Original Train Loss", original_train_loss)
    log_mean_std(model_train._logger, "Masked Train Loss", masked_train_loss)

    log_mean_std(model_train._logger, "Original Test Loss", original_test_loss)
    log_mean_std(model_train._logger, "Masked Test Loss", masked_test_loss)

    log_mean_std(
        model_train._logger, "Paramaters Removed Percentage", parameter_removed_list
    )

    # Now plotting part evolution of sparsification along with training
    # plotting parameters removed after each iteration
    parameter_removed_path = os.path.join(
        model_train.storage_parent_dir, "parameters_removed_list.jpg"
    )
    plot_pruning_evolution(
        parameter_removed_path,
        x_label_data_list,
        parameter_removed_list,
        "Pruning Percentage",
        xlabel="Iteration" if config.step else "Epoch",
    )

    # plotting masked test accuracy
    masked_test_acc_path = os.path.join(
        model_train.storage_parent_dir, "masked_test_acc.jpg"
    )
    plot_pruning_evolution(
        masked_test_acc_path,
        x_label_data_list,
        masked_test_acc,
        "Masked Test Acc",
        xlabel="Iteration" if config.step else "Epoch",
    )

    # plotting masked train accuracy
    masked_train_acc_path = os.path.join(
        model_train.storage_parent_dir, "masked_train_acc.jpg"
    )
    plot_pruning_evolution(
        masked_train_acc_path,
        x_label_data_list,
        masked_train_acc,
        "Masked Train Acc",
        xlabel="Iteration" if config.step else "Epoch" if config.step else "Epoch",
    )

    # plotting original train accuracy
    original_train_acc_path = os.path.join(
        model_train.storage_parent_dir, "original_train_acc.jpg"
    )
    plot_pruning_evolution(
        original_train_acc_path,
        x_label_data_list,
        original_train_acc,
        "Original Train Acc",
        xlabel="Iteration" if config.step else "Epoch",
    )

    # plotting original test accuracy
    original_test_acc_path = os.path.join(
        model_train.storage_parent_dir, "original_test_acc.jpg"
    )
    plot_pruning_evolution(
        original_test_acc_path,
        x_label_data_list,
        original_test_acc,
        "Original Test Acc",
        xlabel="Iteration" if config.step else "Epoch",
    )

    # plotting original vs masked evolution Train
    plot_original_masked(
        x_label_data_list,
        original_result=original_train_acc,
        masked_result=masked_train_acc,
        ylabel="Train Accuracy",
        xlabel="Iteration" if config.step else "Epoch",
        storage_parent_dir=model_train.storage_parent_dir,
    )

    # plotting original vs masked evolution Test
    plot_original_masked(
        x_label_data_list,
        original_result=original_test_acc,
        masked_result=masked_test_acc,
        ylabel="Test Accuracy",
        xlabel="Iteration" if config.step else "Epoch",
        storage_parent_dir=model_train.storage_parent_dir,
    )
