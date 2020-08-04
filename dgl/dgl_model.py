from pytorch_lightning.core import LightningModule
import torch
from training.utils import adjust_learning_rate
from .auxillary_classifier import AuxillaryClassifier
from models.conv_models import ConvModel
from models.model import Model
from custom_pt_layers import Flatten
from layers_modules import pooling_layers
import torch.nn as nn
from .list_module import ListModule


class DglModel(LightningModule):
    """Lightning module used to train decoupled learning based on https://arxiv.org/abs/1901.08164 
    """

    def __init__(
        self, hparams,
    ):
        """initialize lightning module

        Args:
            hparams (namespace): hyper parameters sent to the model
        """
        super(DglModel, self).__init__()
        self.model = hparams.model
        self.data_loaders = {"train": None, "test": None, "val": None}
        self.criterion = None
        self.optimizers = []
        self.n_train_data_per_batch = -1
        self.decay_boundaries = hparams.decay_boundaries
        self.decay_values = hparams.decay_values
        self.up_to_layer_indices = []
        self.sub_models = []
        self.original_indxs = []
        self._init_submodels()

    def _init_submodels(self):
        """initialization and preparation of layer wise auxiliary networks
        """
        current_block = []
        input_tensor = self.model.get_random_input(self.model.input_size).type_as(
            self.model[-1].weight
        )
        prev_conv_indx = -1
        input_shape = [-1, -1, -1, -1]
        origianl_model_indexs = []
        for layer_indx, layer in enumerate(self.model):
            if type(layer) is nn.Conv2d:
                if len(current_block) > 0:
                    aux_model = AuxillaryClassifier(
                        input_features=current_block[0].out_channels,
                        in_size=input_shape[-1],
                        num_classes=self.model.n_output_classes,
                        batchn=self.model.use_batchn,
                    )

                    self.sub_models += [
                        ConvModel(
                            conv_module_list=current_block
                            + aux_model.get_conv_blocks(),
                            linear_module_list=aux_model.get_linear_blocks(),
                            first_linear_output=None,
                            input_size=input_shape[-2:],
                            n_channels=input_shape[1],
                        )
                    ]
                    self.original_indxs += [origianl_model_indexs]

                current_block = []
                origianl_model_indexs = []
                prev_conv_indx = layer_indx
                input_shape = input_tensor.shape
                self.up_to_layer_indices.append(layer_indx)
            elif type(layer) is Flatten:
                break
            input_tensor = layer(input_tensor)
            if type(layer) in pooling_layers.keys():
                continue
            origianl_model_indexs.append(layer_indx)
            current_block.append(layer)
        # last auxiliary
        self.original_indxs += [origianl_model_indexs]
        self.sub_models += [Model(self.model[prev_conv_indx:])]
        self.sub_models = ListModule(*self.sub_models)

    def get_original_model_layers_indx(self, index):
        """get the index of a layer in the model based on the auxiliary network index

        Args:
            index (int): index of the auxiliary network

        Returns:
            int: index of the layer in the original network
        """
        return self.original_indxs[index]

    def __iter__(self):
        """ Returns the Iterator object """
        return iter(self.sub_models)

    def __len__(self):
        return len(self.sub_models)

    def __getitem__(self, index):
        return self.sub_models[index]

    def set_data_loaders(self, data_loaders):
        """sets the data loader used in the training

        Args:
            data_loaders (dict): dictionary holding train, val and test data loaders
        """
        self.data_loaders.update(data_loaders)
        self.n_train_data_per_batch = len(self.data_loaders["train"])

    def train_dataloader(self):
        """returns the train data loader used by pt lightning

        Returns:
            torch.DataLoader: train data loader
        """
        return self.data_loaders["train"]

    def val_dataloader(self):
        """returns the val data loader used by pt lightning

        Returns:
            torch.DataLoader: val data loader
        """
        return self.data_loaders["val"]

    def set_criterion(self, criterion):
        """sets the training criterion

        Args:
            criterion (torch): criterion used as an objective during training
        """
        self.criterion = criterion

    def create_optimizer(self, optim_class, learning_rate):
        """used to create optimizer for auxiliary network

        Args:
            optim_class (torch.optim): optimizer class used for the training
            learning_rate (float): learning rate associated with that optim class
        """
        for model in self.sub_models:
            if optim_class is torch.optim.SGD:
                self.optimizers += [
                    optim_class(
                        model.parameters(),
                        lr=learning_rate,
                        nesterov=True,
                        weight_decay=0.0005,
                        momentum=0.9,
                    )
                ]
            else:
                self.optimizers += [optim_class(self.parameters(), lr=learning_rate)]

    def configure_optimizers(self):
        """returns the list of optimizers for pt lightning

        Returns:
            tuple: list of optimizers and list of scheduler if exists
        """
        return self.optimizers, []

    def training_step(self, batch, batch_idx, optimizer_idx):
        """training step on an auxiliary network of index optimizer_idx

        Args:
            batch (tuple): batch of input data and labels
            batch_idx (int): index of the input batch
            optimizer_idx (int): index of the auxiliary network to be optimized

        Returns:
            dictionary: statistics on the current train step
        """
        x, y = batch
        model = self.sub_models[optimizer_idx]
        for layer_indx in range(self.up_to_layer_indices[optimizer_idx]):
            x = self.model[layer_indx](x)
        y_hat = model.forward(x)
        predictions = y_hat.argmax(dim=1, keepdim=True)
        train_acc = predictions.eq(y.view_as(predictions)).sum().item()
        loss = self.criterion(y_hat, y)
        tensorboard_logs = {"train_loss": loss, "train_acc": train_acc}
        return {
            "loss": loss,
            "log": tensorboard_logs,
            "Acc": train_acc,
        }

    def training_step_end(self, output):
        """called after finishing the training step to adjust optim learning rate

        Args:
            output (list): list of stats of training step

        Returns:
            list: list of input output
        """
        for optimizer in self.optimizers:
            adjust_learning_rate(
                optimizer,
                self.global_step,
                decay_boundaries=self.decay_boundaries,
                decay_values=self.decay_values,
            )
        return output

    def validation_step(self, batch, batch_idx):
        """Validation step called after finishing the training epoch

        Args:
            batch (tuple): tuple of input batch and its associated labels
            batch_idx (int): index of input batch

        Returns:
            dict: stats computed on input validation step
        """
        x, y = batch
        model = self.model
        y_hat = model.forward(x)
        predictions = y_hat.argmax(dim=1, keepdim=True)
        val_loss = self.criterion(y_hat, y)
        val_acc = predictions.eq(y.view_as(predictions)).sum()
        return {"val_loss": val_loss, "val_acc": val_acc, "n_items_batch": y.shape[0]}

    def validation_epoch_end(self, outputs):
        """Aggregates stats from each validation step

        Args:
            outputs (list): list of stats from each validation step call

        Returns:
            dict: aggregated stats for the validation
        """
        if len(outputs) == 0:
            return {"val_loss": -1 * self.global_step}
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_acc"] for x in outputs]).sum() * (
            1 / sum([x["n_items_batch"] for x in outputs])
        )
        tensorboard_logs = {"val_loss": avg_loss, "val_acc": avg_acc}
        return {"val_loss": avg_loss, "val_acc": avg_acc, "log": tensorboard_logs}

    def forward(self, x, index=None):
        """Forward pass on input data x

        Args:
            x (tensor): input data
            index (int, optional): index of auxiliary network to which we are computing the value of its input. Defaults to None.

        Returns:
            tensor: output tensor
        """
        if index is None:
            return self.model.forward(x)
        for layer_indx in range(self.up_to_layer_indices[index]):
            x = self.model[layer_indx](x)
        return x
