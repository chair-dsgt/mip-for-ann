import torch
import os
from custom_pt_layers import pytorch_layers_map
import copy
from .utils import (
    test,
    weights_init,
    device,
    get_confusion_matrix,
    get_precision_recall_curve,
    init_logger,
    Mode,
)
from prettytable import PrettyTable
import time
import numpy as np
from .pl_model import PlModel
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from .logger import LightningProfiler
from argparse import Namespace
import glob
import re
import sys
from dgl import DglModel


class ModelTrain:
    def __init__(
        self,
        model,
        storage_parent_dir,
        input_size=None,
        decay_boundaries=[],
        decay_values=[],
        debug=False,
        finetune_masked=False,
        other_model_parent_dir=None,
        use_cached=True,
        incremental_sparsify=False,
        decoupled_train=False,
    ):
        """initialization of ModelTrain object used to train and test both unmasked and masked model

        Arguments:
            model {models.Model} -- input model object that will be trained and sparsified
            criterion {nn.} -- loss function used during training
            optimizer {torch.optim} -- optimizer used during training
            storage_parent_dir {string} -- parent save directory used to save logs and trained models

        Keyword Arguments:
            input_size {tuple} -- size of input data to the model (default: {None})
            debug {bool} -- when set to true the solver will be verbose (default: {False})
            finetune_masked {bool} -- used to load a sparsified model with its original initialization and finetune it (default: {False})
            other_model_parent_dir {string} -- path of the folder having the model sparsified on another dataset that will be retrained / generalized(default: {None})
            use_cached {boolean} -- boolean to disable loading pretrained model
            incremental_sparsify {boolean} -- boolean to enable incremental sparsify train epoch sparsify then train sparsified model then sparsify etc..
            decoupled_train {boolean} -- flag to enable training using decoupled conv layers
        """
        self._logger = init_logger(storage_parent_dir, debug)
        self.model = model
        self.model_masked = None
        self.pl_model = None
        self.masking_indices = {}
        self.storage_parent_dir = storage_parent_dir
        # used for generalization with retrain masked
        self.other_model_parent_dir = other_model_parent_dir
        self._device = device
        self.input_size = input_size
        self.finetune_masked = finetune_masked
        self.use_cached = use_cached
        self.incremental_sparsify = incremental_sparsify
        self.decay_boundaries = decay_boundaries
        self.decay_values = decay_values
        self.optim_class = None
        self.learning_rate = None
        self.criterion = None
        self.sparsify_masked = False
        self.decoupled_train = decoupled_train
        self.submodule_indx = -1
        self.handlers = {}
        self._init_models()

    def _init_models(self):
        if self.decoupled_train:
            args = {
                "model": self.model,
                "n_train_data_per_batch": -1,
                "decay_boundaries": self.decay_boundaries,
                "decay_values": self.decay_values,
            }
            hparams = Namespace(**args)
            self.pl_model = DglModel(hparams)
        parent_storage_dir = self.storage_parent_dir
        if self.finetune_masked and self.other_model_parent_dir is not None:
            parent_storage_dir = self.other_model_parent_dir
        masked_model_path = os.path.join(
            parent_storage_dir, self.model.name + "_" + Mode.MASK.name + ".pt"
        )
        model_path = self._get_model_ckpt_path()

        # loading masked model initialization if exists
        model_init_path = os.path.join(parent_storage_dir, self.model.name + "_init.pt")
        if os.path.isfile(model_init_path):
            self.load_model_init(model_init_path)
        if self.finetune_masked:
            # now based on the loaded init creating the masked version
            self.swap_pytorch_layers()
            model_masked_loaded = self.load_model_masked(masked_model_path)
            if not (model_masked_loaded):
                self._logger.exception(
                    "Masking indices not loaded for model {}".format(self.model.name)
                )
                raise Exception(
                    "[Exception] Masking indices not set which is a critical bug"
                )
            removed_neurons_percentage = self.swap_pytorch_layers(mask_indices=True)
            self._logger.info(
                "Retraining masked model with {}% removed parameters from original model {}".format(
                    removed_neurons_percentage, self.model.name
                )
            )
        else:
            if model_path is None:
                self.reset_model()
            else:
                self.load_model(model_path)
            self.swap_pytorch_layers()
            if os.path.isfile(masked_model_path):
                if not (self.incremental_sparsify):
                    self.load_model_masked(masked_model_path)

    def len_dgl_submodules(self):
        """returns number of auxiliary networks used in dgl training

        Returns:
            int: number of auxiliary networks
        """
        if self.decoupled_train:
            return len(self.pl_model)
        return 0

    def enable_submodule_dgl(self, indx):
        """enable a specific auxiliary network

        Args:
            indx (int): index of auxiliary network to be enabled for sparsification and testing
        """
        if indx < self.len_dgl_submodules():
            self.submodule_indx = indx

    def get_input_representation(self, input_tensor):
        """computes input representation of input tensor at enabled submodule

        Args:
            input_tensor (tensor): input data

        Returns:
            tensor: the representation to be sent to the auxiliary network
        """
        if self.decoupled_train and self.submodule_indx != -1:
            return self.pl_model.forward(input_tensor, self.submodule_indx)
        return input_tensor

    def get_dgl_layers_to_sparsify(self):
        """get the index of a layer in the model based on the auxiliary network index

        Returns:
            int: index of the layer in the original network
        """
        if self.decoupled_train and self.submodule_indx != -1:
            return self.pl_model.get_original_model_layers_indx(self.submodule_indx)
        return []

    def set_train_params(self, optim_class, learning_rate, criterion):
        """sets the parameters used for training

        Args:
            optim_class (torch.optim): optimizer class
            learning_rate (float): learning rate used
            criterion (torch.criterion): objective used for training
        """
        self.optim_class = optim_class
        self.learning_rate = learning_rate
        self.criterion = criterion

    def get_model_to_sparsify(self):
        """returns the model that needs to be sparsified

        Returns:
            Model: model to be sparsified
        """
        if self.decoupled_train and self.submodule_indx != -1:
            return self.pl_model[self.submodule_indx]
        if self.sparsify_masked:
            return self.model_masked
        if self.incremental_sparsify and self.model_masked is not None:
            return self.model_masked
        return self.model

    def set_model_to_sparsify(self, model):
        """sets the model that needs to be sparsified 
        Args:
            model : pytorch model
        """
        if self.decoupled_train and self.submodule_indx != -1:
            self.pl_model[self.submodule_indx] = model
        if self.sparsify_masked:
            self.model_masked = model
        if self.incremental_sparsify and self.model_masked is not None:
            self.model_masked = model
        self.model = model

    def add_train_listener(self, event_name="epoch", handler=None):
        """add event handler to be called on every training step or epoch
        Arguments:
            handler {function} -- function taking as input sender and a set of args as dict including step number and epoch number
        Keyword Arguments:
            event_name {str} -- can be epoch or step to be called every training step (default: {'epoch'})
        """
        self.handlers[event_name] = handler

    def reset_model(self):
        """Reset model weights
        """
        if self.decoupled_train:
            self.pl_model.apply(weights_init)
        self.model.to(self._device)
        self.model.apply(weights_init)
        self._save_model_init()

    def _get_model_ckpt_path(self):
        resume_from_checkpoint = None
        ckpt_files = glob.glob(self.storage_parent_dir + "/*.ckpt")
        ckpt_files.sort(key=lambda f: int(re.sub("\D", "", f)))
        if len(ckpt_files) > 0 and self.use_cached:
            resume_from_checkpoint = ckpt_files[-1]
        return resume_from_checkpoint

    def load_model(self, path):
        """loads model parameters

        Args:
            path (str): path of the model's checkpoint

        Returns:
            boolean: flag denoting if the operation was successful or not
        """
        try:
            checkpoint = torch.load(path, map_location=self._device)["state_dict"]
            if self.decoupled_train:
                self.pl_model.load_state_dict(checkpoint)
            else:
                state_dict = {}
                for key in checkpoint:
                    if "masked" in key:
                        continue
                    state_dict[re.sub("^model\.", "", key)] = checkpoint[key]
                self.model.load_state_dict(state_dict)
                self.model.eval()
                self._logger.info("Model {} loaded".format(self.model.name))
            return True
        except Exception as e:
            self._logger.exception(str(e))
            return False

    def _save_model_init(self):
        """used to save model initialization

        Returns:
            bool -- return True on success
        """
        try:
            torch.save(
                self.model.state_dict(),
                os.path.join(self.storage_parent_dir, self.model.name + "_init.pt"),
            )
            return True
        except Exception as e:
            self._logger.exception(str(e))
            return False

    def load_model_init(self, path):
        """load saved model initialization from specific path

        Arguments:
            path {string} -- path of the model's initialization

        Returns:
            bool -- return True on success
        """
        try:
            checkpoint = torch.load(path, map_location=self._device)
            self.model.load_state_dict(checkpoint)
            self._logger.info("Model initialization {} loaded".format(self.model.name))
            return True
        except Exception as e:
            self._logger.exception(str(e))
            return False

    def save_model_masked(self, suffix="_masked"):
        """used to save the sparsified model

        Keyword Arguments:
            suffix {str} -- a suffix to be added to the model's name when saved (default: {'_masked'})

        Returns:
            bool -- True on success
        """
        try:
            torch.save(
                {
                    "model_state_dict": self.model_masked.state_dict(),
                    "masking_indices": self.masking_indices,
                },
                os.path.join(self.storage_parent_dir, self.model.name + suffix + ".pt"),
            )
            return True
        except Exception as e:
            self._logger.exception(str(e))
            return False

    def load_model_masked(self, path):
        """load saved masked model

        Arguments:
            path {string} -- path of the masked model's checkpoint

        Returns:
            bool -- True on success
        """
        try:
            checkpoint = torch.load(path, map_location=self._device)
            self.model_masked.load_state_dict(checkpoint["model_state_dict"])
            self.masking_indices = {
                layer: checkpoint["masking_indices"][layer].cpu()
                if torch.is_tensor(checkpoint["masking_indices"][layer])
                else checkpoint["masking_indices"][layer]
                for layer in checkpoint["masking_indices"]
            }
            self._logger.info("Masked Model {} loaded".format(self.model_masked.name))
            self.model_masked.eval()
            return True
        except Exception as e:
            self._logger.exception(str(e))
            return False

    def swap_pytorch_layers(self, mask_indices=False):
        """this function is used to replace dense layers with custom pytorch layers which allows sparsification 
        custom modules are from custom_pt_layers

        Keyword Arguments:
            mask_indices {bool} -- when enabled will mask indices set in ModelTrain object (default: {False})

        Returns:
            float -- parameter removal percentage
        """
        model = self.get_model_to_sparsify()
        model.to(self._device)
        model.eval()
        if not (self.incremental_sparsify and self.model_masked is not None):
            self.model_masked = copy.deepcopy(model)
            self.model_masked.name = self.model.name + "_" + Mode.MASK.name + "_model"
        mask_layer_indx = 0
        removed_params = 0
        original_model_num_params = 1
        if mask_indices:
            original_model_num_params = sum(p.numel() for p in model.parameters())
            removed_params = original_model_num_params
        sample_input = self.model.get_random_input(self.input_size).to(self._device)
        if self.decoupled_train and self.submodule_indx > 0:
            sample_input = self.pl_model.forward(sample_input, self.submodule_indx)
        prev_input_size = sample_input.shape[-2:]
        for layer_indx, layer in enumerate(model):
            # checking for available keys that can be swapped in pytorch layers map
            if (
                type(layer) in pytorch_layers_map.values()
                or type(layer) in pytorch_layers_map
            ):
                masked_layer_class = (
                    pytorch_layers_map[type(layer)]
                    if type(layer) in pytorch_layers_map
                    else type(layer)
                )
                self.model_masked[layer_indx] = masked_layer_class.copy_layer(
                    layer, prev_input_size
                )
            if mask_indices and layer_indx < len(self.model) - 1:
                if (
                    layer_indx in self.masking_indices
                    and len(self.masking_indices[layer_indx]) > 0
                ):
                    self.model_masked[layer_indx].mask_neurons(
                        self.masking_indices[layer_indx]
                    )
                    removed_params -= self.model_masked[
                        layer_indx
                    ].get_sparsified_param_size(self.masking_indices[layer_indx])
                    mask_layer_indx += 1
            sample_input = layer(sample_input)
            prev_input_size = [
                sample_input.shape[i] for i in range(0, len(sample_input.shape))
            ]
            if len(prev_input_size) > 3:
                prev_input_size = prev_input_size[-2:]
            else:
                prev_input_size = prev_input_size[-1]
        self._logger.info(
            "Finished creating masked version of model {}".format(self.model.name)
        )
        return 100 - ((removed_params / original_model_num_params) * 100)

    def set_mask_indices(
        self, mask_indices, suffix="_masked", save_masking_model=False
    ):
        """setting masked indices of the parameters that will be sparsified from the model

        Arguments:
            mask_indices {list} -- a dictionary mapping layer index to a numpy array holding indexes of neurons to be sparsified

        Keyword Arguments:
            suffix {str} -- suffix used to save the masked model (default: {'_masked'})
            save_masking_model {bool} -- when set to true the masked model will be saved (default: {False})
        """
        self.masking_indices = mask_indices
        # create a new model copy
        self.swap_pytorch_layers(True)
        if save_masking_model:
            self.save_model_masked(suffix)

    def copy_masked_model_weights(self):
        for layer_indx, layer_masked in enumerate(self.model_masked):
            if type(layer_masked) in pytorch_layers_map.values():
                self.model[layer_indx] = copy.copy(layer_masked.get_layer())

    def train(
        self, train_loader, val_loader=None, num_epochs=10, finetune_masked=False
    ):
        """train a model based on input train data loader and validation data loader

        Arguments:
            train_loader {torch.dataloader} -- a batch generator for training data

        Keyword Arguments:
            val_loader {torch.dataloader} -- a batch generator for validation data (default: {None})
            num_epochs {int} -- number of training epochs (default: {10})
            finetune_masked {bool} -- when set to True this function will finetune the masked model (default: {False})
        """
        resume_from_checkpoint = self._get_model_ckpt_path()
        train_mode = "training"

        model_checkpoint = ModelCheckpoint(
            filepath=self.storage_parent_dir, monitor="val_loss", mode="min"
        )
        if self.finetune_masked or finetune_masked:
            if self.incremental_sparsify:
                self._logger.error(
                    "Fine tuning of masked model is set to true along with incrementation sparsification"
                )
                return
            train_mode = " masked finetuning"
            model_checkpoint = None
            resume_from_checkpoint = None
            self.model_masked.train()
            for param in self.model_masked.parameters():
                param.requires_grad = True
            self.model_masked.register_backward_hooks()
        if self.incremental_sparsify:
            train_mode = "incremental training masked and original"

        args = {
            "model": self.model,
            "masked_model": self.model_masked,
            "finetune_masked": finetune_masked or self.finetune_masked,
            "incremental_train": self.incremental_sparsify,
            "n_train_data_per_batch": len(train_loader),
            "decay_boundaries": self.decay_boundaries,
            "decay_values": self.decay_values,
        }
        hparams = Namespace(**args)
        if self.decoupled_train and not (self.finetune_masked or finetune_masked):
            pl_model = self.pl_model
        else:
            pl_model = PlModel(hparams)
        pl_model.set_data_loaders({"train": train_loader, "val": val_loader})
        if not (self.decoupled_train):
            for event_name in self.handlers:
                pl_model.add_train_listener(event_name, self.handlers[event_name])
        pl_model.set_criterion(self.criterion)
        pl_model.create_optimizer(self.optim_class, self.learning_rate)
        tensorboard_logger = TensorBoardLogger(
            self.storage_parent_dir, name=self.model.name
        )
        profiler = LightningProfiler(self._logger)
        model_trainer = Trainer(
            profiler=profiler,
            progress_bar_refresh_rate=0,
            logger=tensorboard_logger,
            checkpoint_callback=model_checkpoint,
            resume_from_checkpoint=resume_from_checkpoint,
            max_epochs=num_epochs,
            gpus=1 if torch.cuda.is_available() else None,
            check_val_every_n_epoch=1,
            early_stop_callback=None,
        )

        self._logger.info("Started Model {} {}".format(self.model.name, train_mode))
        model_trainer.fit(pl_model)
        pl_model.freeze()
        self.model_masked.unregister_backward_hooks()
        self.model_masked.eval()
        # self.model_masked.unregister_backward_hooks()
        self.model.eval()
        self._logger.info("Finished Model {} training".format(self.model.name))

    def print_results(
        self,
        train_loader,
        val_loader,
        test_loader,
        test_original_model=True,
        test_masked_model=True,
        save_heat_map=False,
        mode_name="",
        log_results=True,
    ):
        """used to print accuracy/loss results of the original and masked model on input datasets

        Arguments:
            train_loader {torch.dataloader} -- train data loader
            val_loader {torch.dataloader} -- validation data loader
            test_loader {torch.dataloader} -- test data loader

        Keyword Arguments:
            test_original_model {bool} -- a flag to test the original model (default: {True})
            test_masked_model {bool} -- a flag to test the masked model (default: {True})
            save_heat_map {bool} -- a flag to save the model heatmap on random images for conv models (default: {False})
            mode_name {str} -- name of the sparsification mode that can be MASK, CRITICAL, RANDOM (default: {''})
            log_results {bool} -- flag to just return a list of model results without writing it to the logs

        Returns:
            string -- result table 
        """
        if not (test_masked_model) and not (test_original_model):
            self._logger.exception("Error when testing model no model is enabled")
            return
        col_names = ["Model Name"]
        if train_loader is not None:
            col_names.append("Train l/acc")
        if val_loader is not None:
            col_names.append("Val l/acc")
        if test_loader is not None:
            col_names.append("Test l/acc")
        model_summary_table = PrettyTable(col_names)
        model_list = []
        if test_original_model and not (self.finetune_masked):
            model_list = [
                ("Original Model " + self.model.name, self.model.to(self._device))
            ]
        if test_masked_model:
            if self.model_masked is not None:
                model_list.append(
                    (
                        "Masked Model " + self.model_masked.name,
                        self.model_masked.to(self._device),
                    )
                )
        # saving heat map on some sample images
        if val_loader is not None:
            if save_heat_map:
                for label in range(val_loader.dataset.n_classes):
                    X, y = val_loader.dataset.sample_itm_class(label)
                    for model in model_list:
                        if hasattr(model[-1], "get_heat_map"):
                            # sample an image for class 0 for now
                            img_path = model[-1].get_heat_map(
                                X.clone(),
                                y,
                                self.storage_parent_dir,
                                model[0] + "_" + mode_name,
                            )
                            self._logger.info("Created Heat map at {}".format(img_path))
            for model in model_list:
                confusion_matrix_path = os.path.join(
                    self.storage_parent_dir,
                    "confusion_matrix_" + model[0] + "_" + mode_name + ".png",
                )
                get_confusion_matrix(model[-1], val_loader, confusion_matrix_path)
                precision_recall_path = os.path.join(
                    self.storage_parent_dir,
                    "cprecision_recall_" + model[0] + "_" + mode_name + ".png",
                )
                get_precision_recall_curve(model[-1], val_loader, precision_recall_path)
                self._logger.info(
                    "Created Confusion matrix at {}".format(confusion_matrix_path)
                )
        results = []
        for model in model_list:
            current_model_results_dict = {}
            loss_acc = [0] * (len(col_names) - 1)
            start_indx = 0
            if train_loader is not None:
                loss_acc[start_indx] = test(model[-1], train_loader, self.criterion)
                current_model_results_dict["loss_train"] = loss_acc[start_indx][0]
                current_model_results_dict["acc_train"] = loss_acc[start_indx][1]
                start_indx += 1
            if val_loader is not None:
                loss_acc[start_indx] = test(model[-1], val_loader, self.criterion)
                start_indx += 1
            if test_loader is not None:
                loss_acc[start_indx] = test(model[-1], test_loader, self.criterion)
                current_model_results_dict["loss_test"] = loss_acc[start_indx][0]
                current_model_results_dict["acc_test"] = loss_acc[start_indx][1]
                start_indx += 1
            resulting_info = [
                str(current_l_acc[0]) + " / " + str(current_l_acc[1])
                for current_l_acc in loss_acc
            ]
            model_summary_table.add_row([model[0]] + resulting_info)
            results.append(current_model_results_dict)
        if log_results:
            self._logger.info("[Results]")
            self._logger.info(str(model_summary_table))
        return results
