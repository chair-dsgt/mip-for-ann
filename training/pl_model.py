from pytorch_lightning.core import LightningModule
import torch
from .utils import adjust_learning_rate


class PlModel(LightningModule):
    def __init__(
        self, hparams,
    ):
        super(PlModel, self).__init__()
        self.model = hparams.model
        self.masked_model = hparams.masked_model
        self.data_loaders = {"train": None, "test": None, "val": None}
        self.finetune_masked = hparams.finetune_masked
        self.incremental_train = hparams.incremental_train
        self.criterion = None
        self.optimizers = []
        self.handlers = {}
        self.n_train_data_per_batch = -1
        self.decay_boundaries = hparams.decay_boundaries
        self.decay_values = hparams.decay_values
        self.optimizer_idx = None
        self._called_handler = False
        self.cached_batch_x = None

    def add_train_listener(self, event_name="epoch", handler=None):
        """add event handler to be called on every training step or epoch
        Arguments:
            handler {function} -- function taking as input sender and a set of args as dict including step number and epoch number
        Keyword Arguments:
            event_name {str} -- can be epoch or step to be called every training step (default: {'epoch'})
        """
        self.handlers[event_name] = handler

    def clear_handlers(self):
        self.handlers = {}

    def set_data_loaders(self, data_loaders):
        """sets the data loader used in the training

        Args:
            data_loaders (dict): dictionary holding train, val and test data loaders
        """
        self.data_loaders.update(data_loaders)
        self.cached_batch_x = next(iter(data_loaders["train"]))[0]
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
        for model in [self.model, self.masked_model]:
            if model is None:
                continue
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
        if self.incremental_train:
            return self.optimizers, []
        elif self.finetune_masked:
            return self.optimizers[1]
        else:
            return self.optimizers[0]

    def _select_model(self, optimizer_idx=None):
        if optimizer_idx is None:
            if self.finetune_masked:
                return self.masked_model, self.optimizers[1]
            return self.model, self.optimizers[0]
        elif optimizer_idx == 1:
            return self.masked_model, self.optimizers[1]
        return self.model, self.optimizers[0]

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        """training step on the model to be trained

        Args:
            batch (tuple): batch of input data and labels
            batch_idx (int): index of the input batch
            optimizer_idx (int): index of the auxiliary network to be optimized

        Returns:
            dictionary: statistics on the current train step
        """
        x, y = batch
        model, optimizer = self._select_model(optimizer_idx)
        for param in model.parameters():
            param.requires_grad = True
        y_hat = model.forward(x)
        predictions = y_hat.argmax(dim=1, keepdim=True)
        train_acc = predictions.eq(y.view_as(predictions)).sum().item()
        loss = self.criterion(y_hat, y)
        tensorboard_logs = {"train_loss": loss, "train_acc": train_acc}
        self.optimizer_idx = optimizer_idx
        return {
            "loss": loss,
            "log": tensorboard_logs,
            "Acc": train_acc,
        }

    def _call_handler(self, call_handler=True):
        epoch_indx = self.current_epoch
        step_number = self.global_step
        batch_idx = step_number % self.n_train_data_per_batch
        if step_number == 0 and epoch_indx == 0 and self.incremental_train:
            return False
        if "step" in self.handlers and self.handlers["step"] is not None:
            if call_handler:
                self.handlers["step"]({"epoch": epoch_indx, "step": step_number})
            return True
        if (
            batch_idx == 0
            and "epoch" in self.handlers
            and self.handlers["epoch"] is not None
        ):
            if call_handler:
                self.handlers["epoch"]({"epoch": epoch_indx})
            return True
        return False

    def training_step_end(self, output):
        """called after finishing the training step to adjust optim learning rate

        Args:
            output (list): list of stats of training step

        Returns:
            list: list of input output
        """
        model, optimizer = self._select_model(self.optimizer_idx)
        if self._call_handler():
            self._called_handler = True
        adjust_learning_rate(
            optimizer,
            self.global_step,
            decay_boundaries=self.decay_boundaries,
            decay_values=self.decay_values,
        )
        return output

    def validation_step(self, batch, batch_idx, optimizer_idx=None):
        """Validation step called after finishing the training epoch

        Args:
            batch (tuple): tuple of input batch and its associated labels
            batch_idx (int): index of input batch

        Returns:
            dict: stats computed on input validation step
        """
        x, y = batch
        model, _ = self._select_model(optimizer_idx)
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
            return {
                "val_loss": torch.tensor(-1 * self.global_step).type_as(
                    self.cached_batch_x
                )
            }
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_acc"] for x in outputs]).sum() * (
            1 / sum([x["n_items_batch"] for x in outputs])
        )
        tensorboard_logs = {"val_loss": avg_loss, "val_acc": avg_acc}
        return {"val_loss": avg_loss, "val_acc": avg_acc, "log": tensorboard_logs}

    def forward(self, x):
        raise NotImplementedError
