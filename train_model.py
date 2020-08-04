from models import (
    ConvModel,
    FullyConnectedBaselineModel,
    FullyConnected2Model,
    FullyConnected3Model,
    FullyConnected4Model,
    ConvBaselineModel,
    VGG19,
    VGG16,
    VGG16NoBn,
    VGG7,
)
from training import ModelTrain
import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from dataset import BalancedDataLoader, Caltech256, TinyImageNet
from training import device, get_storage_dir, log_config
import copy
import multiprocessing
import sys
from glob import glob
import numpy as np

"""Script used to train a model based on input configuration
"""

model_indx_map = {
    0: FullyConnectedBaselineModel,
    1: FullyConnected2Model,
    2: FullyConnected3Model,
    3: FullyConnected4Model,
    4: ConvBaselineModel,
    5: VGG7,
    6: VGG16,
    7: VGG16NoBn,
    8: VGG19,
}

model_indx_is_conv = {}
for model_indx in model_indx_map:
    model_indx_is_conv[model_indx] = issubclass(model_indx_map[model_indx], ConvModel)

dataset_map = {
    0: datasets.MNIST,
    1: datasets.FashionMNIST,
    2: datasets.KMNIST,
    3: Caltech256,
    4: datasets.CIFAR10,
    5: datasets.CIFAR100,
    6: TinyImageNet,
}

learning_rates = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
optimizers = [torch.optim.Adam, torch.optim.SGD, torch.optim.RMSprop]


def process_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", "-m", default=4, type=int)
    parser.add_argument("--epochs", "-e", default=1, type=int)
    parser.add_argument("--batch-size", "-bs", default=100, type=int)
    parser.add_argument("--learning-rate", "-l", default=2, type=int)
    parser.add_argument(
        "--decay_boundaries",
        nargs="+",
        type=int,
        default=[],
        help="boundaries for piecewise_constant decay",
    )
    parser.add_argument(
        "--decay_values",
        nargs="+",
        type=float,
        default=[],
        help="values for piecewise_constant decay",
    )
    parser.add_argument("--image-size", "-imsize", default=32, type=int)
    parser.add_argument("--storage-dir", "-sd", default=None)
    parser.add_argument("--other-model-dir", "-omd", default=None)
    # used to train multiple models with different initializations
    parser.add_argument("--num-resets", "-r", default=1, type=int)
    parser.add_argument("--debug", "-d", default=True, type=bool)
    parser.add_argument("--dataset", "-dl", default=0, type=int)
    parser.add_argument("--optimizer", "-op", default=0, type=int)
    parser.add_argument("--retrain", "-rt", action="store_true")
    parser.add_argument("--decoupled-train", "-dgl", action="store_true")
    return parser


def prepare_model(config, input_size=784, n_channels=1, n_output_classes=10):
    if config.storage_dir is None:
        raise Exception(
            "[Exception] Please provide storage-dir argument to save model data"
        )
    if config.model in model_indx_map:
        model = model_indx_map[config.model](
            input_size=input_size,
            n_channels=n_channels,
            n_output_classes=n_output_classes,
        ).to(device)
    else:
        raise Exception(
            "[Exception] Model indx {} is not right".format(str(config.model))
        )
    return model


def prepare_model_train(
    config,
    input_size,
    n_channels=1,
    n_output_classes=10,
    exp_indx=0,
    prefix="",
    use_cached=True,
    incremental_sparsify=False,
    data_loaders=None,
    other_model_parent_dir=None,
):
    model = prepare_model(
        config,
        input_size=input_size,
        n_channels=n_channels,
        n_output_classes=n_output_classes,
    )
    criterion = torch.nn.CrossEntropyLoss()
    learning_rate = learning_rates[config.learning_rate]
    optim_class = optimizers[config.optimizer]
    storage_parent_dir = get_storage_dir(config, model.name, exp_indx, prefix)
    model_train = ModelTrain(
        model,
        storage_parent_dir,
        debug=config.debug,
        input_size=input_size,
        decay_boundaries=config.decay_boundaries,
        decay_values=config.decay_values,
        finetune_masked=config.retrain,
        other_model_parent_dir=other_model_parent_dir,
        use_cached=use_cached,
        incremental_sparsify=incremental_sparsify,
        decoupled_train=config.decoupled_train,
    )
    model_train.set_train_params(optim_class, learning_rate, criterion)
    return model_train


def prepare_dataset(config, is_color=False):
    if config.dataset in dataset_map:
        transformation_list = [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.Grayscale(3),
        ]

        transformation_list.append(transforms.ToTensor())
        dataset_transform = transforms.Compose(transformation_list)
        test_dataset = dataset_map[config.dataset](
            "../data", train=False, download=True, transform=dataset_transform
        )
        random_crop_size = next(iter(test_dataset))[0].shape[-1]
        train_transformation = []
        if config.dataset > 2:
            train_transformation += [
                transforms.RandomCrop(random_crop_size, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        train_transformation += transformation_list
        train_dataset_transform = transforms.Compose(train_transformation)
        train_dataset = dataset_map[config.dataset](
            "../data", train=True, download=True, transform=train_dataset_transform
        )
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        val_dataset = copy.deepcopy(val_dataset)
        val_dataset.dataset = dataset_map[config.dataset](
            "../data", train=True, download=True, transform=dataset_transform
        )
        num_workers = multiprocessing.cpu_count() - 1
        gettrace = getattr(sys, "gettrace", None)
        if gettrace and gettrace():
            num_workers = 0
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        val_dataset = BalancedDataLoader(
            train_dataset.dataset, selected_indices=val_dataset.indices
        )
        # as the validation is used for MIp and needs to be balanced
        val_loader = DataLoader(
            val_dataset, batch_size=100, shuffle=False, num_workers=num_workers
        )
        test_loader = DataLoader(
            test_dataset, batch_size=100, shuffle=False, num_workers=num_workers
        )
    else:
        raise Exception(
            "[Exception] Dataset indx {} is not right".format(str(config.dataset))
        )
    data_loaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    return data_loaders


if __name__ == "__main__":
    parser = process_args()
    config = parser.parse_args()
    data_loaders = prepare_dataset(config)
    X, _ = next(iter(data_loaders["val"]))
    if model_indx_is_conv[config.model]:
        # conv model
        input_size = X.shape[-2:]
    else:
        input_size = X.flatten().cpu().numpy().reshape(X.shape[0], -1).shape[-1]
    n_channels = X.shape[1]
    n_output_classes = len(data_loaders["train"].dataset.dataset.class_to_idx.values())
    n_experiments = config.num_resets
    exp_dirs = [None for _ in range(n_experiments)]
    model_list = ["original", "Masked"]
    if config.other_model_dir is not None:
        exp_dirs = glob(config.other_model_dir.rstrip("/") + "/exp_*/")
        n_experiments = len(exp_dirs)
        model_list = ["Generalization"]
    all_exp_results = {}
    for exp_indx in range(n_experiments):
        model_train = prepare_model_train(
            config,
            input_size,
            n_channels=n_channels,
            n_output_classes=n_output_classes,
            exp_indx=exp_indx,
            data_loaders=data_loaders,
            other_model_parent_dir=exp_dirs[exp_indx],
        )
        log_config(model_train._logger, config)
        model_train.train(
            data_loaders["train"],
            val_loader=data_loaders["val"],
            num_epochs=config.epochs,
        )
        model_results = model_train.print_results(
            data_loaders["train"], data_loaders["val"], data_loaders["test"],
        )
        for model_indx, mode_name in enumerate(model_list):
            for metric_name in model_results[model_indx]:
                key_results_name = mode_name + metric_name
                if key_results_name not in all_exp_results:
                    all_exp_results[key_results_name] = []
                all_exp_results[key_results_name].append(
                    model_results[model_indx][metric_name]
                )
        if config.retrain and config.other_model_dir is not None:
            model_train._logger.info(
                "Finished Generalization #{} from {} for Model {}".format(
                    exp_indx, config.other_model_dir, model_train.model.name
                )
            )
    if n_experiments > 1:
        for model_name in model_list:
            for metric_name in [
                "loss_train",
                "acc_train",
                "loss_test",
                "acc_test",
            ]:
                results_list = all_exp_results[model_name + metric_name]
                metric_name_clean = " ".join(metric_name.split("_")).capitalize()
                model_train._logger.info(
                    "{} {} mean {} +- {}".format(
                        model_name,
                        metric_name_clean,
                        np.mean(results_list),
                        np.std(results_list),
                    )
                )
