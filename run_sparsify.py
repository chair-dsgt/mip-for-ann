import os
from train_model import (
    process_args,
    prepare_model_train,
    prepare_dataset,
    prepare_model,
    model_indx_is_conv,
)
from dataset import MIPBatchLoader
from sparsify import SparsifyModel, SparsifySequential, SparsifyBackward, SparsifyDgl
import numpy as np
from training.utils import get_storage_dir, log_config, Mode
import time

"""script used to sparsify a trained model by computing the importance score of each neuron and pruning non-critical neurons
"""


def prepare_arguments():
    parser = process_args()
    parser.add_argument("--num-samples", "-n", default=10, type=int)
    parser.add_argument("--sparsification-weight", "-sw", default=5, type=int)
    parser.add_argument("--threshold", "-tt", default=1e-1, type=float)
    parser.add_argument("--mean-threshold", "-mth", action="store_true")
    parser.add_argument("--force", "-f", action="store_true")
    parser.add_argument("--heat-map", "-hm", action="store_true")
    parser.add_argument("--fine-tune", "-ft", action="store_true")
    parser.add_argument("--sequential-run", "-seq", action="store_true")
    parser.add_argument("--relaxed", "-rl", action="store_true")
    parser.add_argument("--tuned-epochs", "-te", default=1, type=int)
    parser.add_argument("--prune-from", "-pf", type=int)
    parser.add_argument("--layer-by-layer", "-bll", action="store_true")
    return parser


def prepare_config():
    parser = prepare_arguments()
    config = parser.parse_args()
    return config


def prepare_images_mip_input(mip_batch_loader):
    X, y, initial_bounds = next(mip_batch_loader)
    input_size, n_channels = mip_batch_loader.get_input_n_channels()
    n_output_classes = mip_batch_loader.get_n_output_classes()
    return X, y, initial_bounds, input_size, n_output_classes, n_channels


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
    exp_indx = 0
    all_exp_results = {}
    finetuning_prefix = "Fine Tuned"
    pruning_percentage_prefix = "Pruning Percentage"
    tuning_epochs = config.tuned_epochs

    sparsification_time_list = []
    model_train = None
    while True:
        model = prepare_model(
            config,
            input_size=input_size,
            n_channels=n_channels,
            n_output_classes=n_output_classes,
        )
        storage_parent_dir = get_storage_dir(config, model.name, exp_indx)
        del model
        if not (os.path.isdir(storage_parent_dir)):
            break
        model_train = prepare_model_train(
            config,
            input_size,
            n_channels=n_channels,
            n_output_classes=n_output_classes,
            exp_indx=exp_indx,
        )
        log_config(model_train._logger, config)
        mip_data_loader.set_model(
            model_train.model
        )  # evaluator is accuracy computation the default one
        X, y, initial_bounds = next(mip_data_loader)
        if config.sequential_run:
            # running independently on each class then taking the average
            sparsify = SparsifySequential(
                model_train,
                mip_data_loader,
                threshold=config.threshold,
                sparsification_weight=config.sparsification_weight,
                relaxed_constraints=config.relaxed,
                n_output_classes=n_output_classes,
                mean_threshold=config.mean_threshold,
            )
        else:
            sparsify = SparsifyModel(
                model_train,
                threshold=config.threshold,
                sparsification_weight=config.sparsification_weight,
                relaxed_constraints=config.relaxed,
                mean_threshold=config.mean_threshold,
            )
            if not (config.decoupled_train):
                sparsify.create_bounds(initial_bounds)

        if config.layer_by_layer:
            sparsify = SparsifyBackward(
                sparsify, mip_data_loader, n_output_classes=n_output_classes,
            )
        elif config.decoupled_train:
            sparsify = SparsifyDgl(
                sparsify, mip_data_loader, n_output_classes=n_output_classes,
            )
        parameters_removed_percentage = 0
        for mode in [Mode.MASK, Mode.Random, Mode.CRITICAL]:
            start_sparsify_time = time.time()
            removal_percentage = sparsify.sparsify_model(
                X,
                y,
                mode=mode,
                use_cached=not (config.force),
                start_pruning_from=config.prune_from,
            )
            sparsification_time = time.time() - start_sparsify_time
            if mode == Mode.MASK:
                parameters_removed_percentage = removal_percentage
                sparsification_time_list.append(sparsification_time)
            model_results = model_train.print_results(
                train_loader,
                val_loader,
                test_loader,
                save_heat_map=config.heat_map,
                mode_name=mode.name,
            )

            model_results[-1][pruning_percentage_prefix] = parameters_removed_percentage
            for model_indx, mode_name in enumerate(["original", mode.name]):
                if (pruning_percentage_prefix) not in model_results[model_indx]:
                    model_results[model_indx][pruning_percentage_prefix] = 0
                for metric_name in model_results[model_indx]:
                    key_results_name = mode_name + metric_name
                    if key_results_name not in all_exp_results:
                        all_exp_results[key_results_name] = []
                    all_exp_results[key_results_name].append(
                        model_results[model_indx][metric_name]
                    )
            # Fine tune all modes to compare results
            if config.fine_tune and tuning_epochs > 0:
                model_train.train(
                    train_loader,
                    val_loader=None,
                    num_epochs=tuning_epochs,
                    finetune_masked=True,
                )
                finetuned_model_results = model_train.print_results(
                    train_loader,
                    val_loader,
                    test_loader,
                    mode_name=mode.name,
                    test_original_model=False,
                )
                finetuned_model_results[0][
                    pruning_percentage_prefix
                ] = parameters_removed_percentage
                for metric_name in finetuned_model_results[0]:
                    key_results_name = finetuning_prefix + mode.name + metric_name
                    if key_results_name not in all_exp_results:
                        all_exp_results[key_results_name] = []
                    all_exp_results[key_results_name].append(
                        finetuned_model_results[0][metric_name]
                    )
        sparsify.reset()
        exp_indx += 1
    # Now logging mean and variance of sparsification results
    if exp_indx > 0:
        list_modes = ["original", Mode.MASK.name, Mode.Random.name, Mode.CRITICAL.name]
        if config.fine_tune:
            list_modes.insert(2, finetuning_prefix + Mode.MASK.name)
            list_modes.insert(3, finetuning_prefix + Mode.Random.name)
            list_modes.insert(4, finetuning_prefix + Mode.CRITICAL.name)
        for mode in list_modes:
            for metric_name in [
                "loss_train",
                "acc_train",
                "loss_test",
                "acc_test",
                pruning_percentage_prefix,
            ]:
                results_list = all_exp_results[mode + metric_name]
                metric_name_clean = " ".join(metric_name.split("_")).capitalize()
                model_train._logger.info(
                    "{} {} mean {} +- {}".format(
                        mode,
                        metric_name_clean,
                        np.mean(results_list),
                        np.std(results_list),
                    )
                )
    if model_train is not None:
        model_train._logger.info(
            "Sparsification time including swapping pytorch layers mean {} +- {}".format(
                np.mean(sparsification_time_list), np.std(sparsification_time_list)
            )
        )
