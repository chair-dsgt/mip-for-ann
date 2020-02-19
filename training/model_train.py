from .logger import Logger
import torch
import os
from custom_pt_layers import pytorch_layers_map
from custom_pt_layers import MaskedLinear
from layers_modules import layers_modules_maps
import copy
from .utils import test, weights_init, device, Mode, adjust_learning_rate
from prettytable import PrettyTable
import time
import numpy as np


class ModelTrain:
    def __init__(self, model, criterion, optimizer, storage_parent_dir, input_size=None, debug=False, retrain_masked=False, other_model_parent_dir=None):
        """initialization of ModelTrain object used to train and test both unmasked and masked model

        Arguments:
            model {models.Model} -- input model object that will be trained and sparsified
            criterion {nn.} -- loss function used during training
            optimizer {torch.optim} -- optimizer used during training
            storage_parent_dir {string} -- parent save directory used to save logs and trained models

        Keyword Arguments:
            input_size {tuple} -- size of input data to the model (default: {None})
            debug {bool} -- when set to true the solver will be verbose (default: {False})
            retrain_masked {bool} -- used to load a sparsified model with its original initialization and retrain it (default: {False})
            other_model_parent_dir {string} -- path of the folder having the model sparsified on another dataset that will be retrained / generalized(default: {None})
        """
        self._logger = Logger.__call__(storage_parent_dir, debug).get_logger()
        self.model = model
        self.model_masked = None
        self.masking_indices = []
        self.criterion = criterion
        self.optimizer = optimizer
        self.epoch = 0
        self.storage_parent_dir = storage_parent_dir
        # usef for generalization with retrain masked
        self.other_model_parent_dir = other_model_parent_dir
        self._device = device
        self.input_size = input_size
        self.retrain_masked = retrain_masked
        self._init_models()

    def _init_models(self):
        parent_storage_dir = self.storage_parent_dir
        if self.retrain_masked and self.other_model_parent_dir is not None:
            parent_storage_dir = self.other_model_parent_dir
        masked_model_path = os.path.join(
            parent_storage_dir, self.model.name+'_'+Mode.MASK.name+'.pt')
        model_path = os.path.join(
            self.storage_parent_dir, self.model.name+'.pt')
        if self.retrain_masked:
            # loading masked model initialization and setting layer importance
            model_init_path = os.path.join(
                parent_storage_dir, self.model.name+'_init.pt')
            self.load_model_init(model_init_path)
            # now based on the loaded init creating the masked version
            self.swap_pytorch_layers()
            model_masked_loaded = self.load_model_masked(masked_model_path)
            if not(model_masked_loaded):
                self._logger.exception(
                    'Masking indices not loaded for model {}'.format(self.model.name))
                raise Exception(
                    'Masking indices not set which is a critical bug')
            removed_neurons_percentage = self.swap_pytorch_layers(
                mask_indices=True)
            self._logger.info('Retraining masked model with {}% removed parameters from original model {}'.format(
                removed_neurons_percentage, self.model.name))
        else:
            if not(os.path.isfile(model_path)):
                self.reset_model()
            if os.path.isfile(masked_model_path):
                self.swap_pytorch_layers()
                self.load_model_masked(masked_model_path)
        if os.path.isfile(model_path):
            self.load_model(model_path)

    def reset_model(self):
        """Reset model weights
        """   
        self.model.to(self._device)
        self.model.apply(weights_init)
        self._save_model_init()

    def _save_model_init(self):
        """used to save model initialization
        
        Returns:
            bool -- return True on success
        """        
        try:
            torch.save(self.model.state_dict(), os.path.join(
                self.storage_parent_dir, self.model.name + '_init.pt'))
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
            self._logger.info(
                'Model initialization {} loaded'.format(self.model.name))
            return True
        except Exception as e:
            self._logger.exception(str(e))
            return False

    def save_model(self, model, optimizer, prefix=''):
        """save model state dict along with optimizer state to continue training on interruptions
        
        Arguments:
            model {models.Model} -- model used for training
            optimizer {torch.optim} -- optimizer used during training
        
        Keyword Arguments:
            prefix {str} -- prefix added during model save (default: {''})
        
        Returns:
            bool -- True on success
        """        
        try:
            torch.save({
                'epoch': self.epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(self.storage_parent_dir, model.name+prefix+'.pt'))
            self._logger.info('Model {} saved at epoch {}'.format(
                self.model.name, str(self.epoch)))
            return True
        except Exception as e:
            self._logger.exception(str(e))
            return False

    def load_model(self, path):
        """used to load model from specific path
        
        Arguments:
            path {string} -- path of the checkpoint holding model and optimizer state dict
        
        Returns:
            bool -- True on success
        """        
        try:
            checkpoint = torch.load(path, map_location=self._device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if self.optimizer is not None:
                self.optimizer.load_state_dict(
                    checkpoint['optimizer_state_dict'])
            self.epoch = checkpoint['epoch']
            self._logger.info('Model {} loaded'.format(self.model.name))
            return True
        except Exception as e:
            self._logger.exception(str(e))
            return False

    def save_model_masked(self, suffix='_masked'):
        """used to save the sparsified model
        
        Keyword Arguments:
            suffix {str} -- a suffix to be added to the model's name when saved (default: {'_masked'})
        
        Returns:
            bool -- True on success
        """        
        try:
            torch.save({
                'model_state_dict': self.model_masked.state_dict(),
                'masking_indices': self.masking_indices
            }, os.path.join(self.storage_parent_dir, self.model.name + suffix + '.pt'))
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
            self.model_masked.load_state_dict(checkpoint['model_state_dict'])
            self.masking_indices = checkpoint['masking_indices']
            self._logger.info(
                'Masked Model {} loaded'.format(self.model_masked.name))
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
        self.model_masked = copy.deepcopy(self.model)
        self.model_masked.name = self.model.name + '_' + Mode.MASK.name + '_model'
        mask_layer_indx = 0
        prev_input_size = self.input_size
        prev_layer_indx = 0
        first_layer_checked = False
        removed_params = 0
        original_model_num_params = 1
        if mask_indices:
            original_model_num_params = sum(
                p.numel() for p in self.model.parameters())
            removed_params = original_model_num_params
        for layer_indx, layer in enumerate(self.model):
            # checking for available keys that can be swapped in pytorch layers map
            for layer_type in pytorch_layers_map:
                if isinstance(layer, layer_type):
                    # needs swapping in order to update image size
                    if first_layer_checked:
                        prev_input_size = self.model_masked[prev_layer_indx].output_size
                    self.model_masked[layer_indx] = pytorch_layers_map[layer_type].copy_layer(
                        layer, prev_input_size)
                    if mask_indices and len(self.masking_indices) > 0 and layer_indx < len(self.model) - 1:
                        if self.masking_indices[mask_layer_indx][0] == layer_indx:
                            self.model_masked[layer_indx].mask_neurons(
                                self.masking_indices[mask_layer_indx][1])
                            removed_params -= len(
                                self.masking_indices[mask_layer_indx][1]) * self.model_masked[layer_indx].in_features
                            mask_layer_indx += 1
                    prev_layer_indx = layer_indx
                    first_layer_checked = True
                    break
        self._logger.info(
            'Finished creating masked version of model {}'.format(self.model.name))
        return 100 - ((removed_params / original_model_num_params) * 100)

    def set_mask_indices(self, mask_indices, suffix='_masked', save_masking_model=False):
        """setting masked indices of the parameters that will be sparsified from the model
        
        Arguments:
            mask_indices {list} -- a list of numpy array holding indexes of neurons to be sparsified
        
        Keyword Arguments:
            suffix {str} -- suffix used to save the masked model (default: {'_masked'})
            save_masking_model {bool} -- when set to true the masked model will be saved (default: {False})
        """        
        self.masking_indices = mask_indices
        # create a new model copy
        self.swap_pytorch_layers(True)
        if save_masking_model:
            self.save_model_masked(suffix)

    def train(self, train_loader, val_loader=None, num_epochs=10, finetune_masked=False):
        """train a model based on input train data loader and validation data loader
        
        Arguments:
            train_loader {torch.dataloader} -- a batch generator for training data
        
        Keyword Arguments:
            val_loader {torch.dataloader} -- a batch generator for validation data (default: {None})
            num_epochs {int} -- number of training epochs (default: {10})
            finetune_masked {bool} -- when set to True this function will finetune the masked model (default: {False})
        """        
        model = self.model
        optimizer = self.optimizer
        prefix = ''
        if self.retrain_masked or finetune_masked:
            model = self.model_masked
            optimizer = self.optimizer.__class__(
                model.parameters(), self.optimizer.param_groups[0]['lr'])
            self.epoch = 0
            if finetune_masked:
                prefix = '_finetuned'
        model.train()
        train_batch_time_list = []
        inference_batch_time_list = []
        self._logger.info(
            'Started training Model {} training'.format(model.name))
        for epoch_indx in range(self.epoch, num_epochs):
            total_loss, total_err = 0., 0.
            for X, y in train_loader:
                start_batch_time = time.time()
                X, y = X.to(self._device), y.to(self._device)
                yp = model(X)
                loss = self.criterion(yp, y)
                optimizer.zero_grad()

                model.register_backward_hooks()
                loss.backward()
                optimizer.step()
                train_batch_time_list.append(time.time() - start_batch_time)

                total_err += (yp.max(dim=1)[1] != y).sum().item()
                total_loss += loss.item() * X.shape[0]
            adjust_learning_rate(optimizer, epoch_indx)
            self._logger.logging_loss(
                'train',  total_loss / len(train_loader.dataset), epoch_indx)
            if val_loader is not None:
                model.eval()
                start_eval_time = time.time()
                val_loss, _ = test(model, val_loader, self.criterion)
                inference_batch_time_list.append(
                    (time.time() - start_eval_time) / len(val_loader))
                self._logger.logging_loss('val',  val_loss, epoch_indx)
                model.train()
            self.epoch = epoch_indx
            self.save_model(model, optimizer, prefix=prefix)
        model.eval()
        model.unregister_backward_hooks()
        avg_time_per_batch = np.mean(train_batch_time_list)
        total_train_time = np.sum(train_batch_time_list)
        avg_inference_per_batch = np.mean(inference_batch_time_list)
        self._logger.info('Model {} took {} seconds per batch'.format(
            model.name, avg_time_per_batch))
        self._logger.info('Model {} took total {} seconds to train'.format(
            model.name, total_train_time))
        self._logger.info('Model {} took average  {} seconds for inference per batch'.format(
            model.name, avg_inference_per_batch))
        self._logger.info('Finished Model {} training'.format(model.name))

    def print_results(self, train_loader, val_loader, test_loader, test_original_model=True, test_masked_model=True, save_heat_map=False, mode_name=''):
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
        
        Returns:
            string -- result table 
        """        
        if not(test_masked_model) and not(test_original_model):
            self._logger.exception(
                'Error when testing model no model is enabled')
            return
        col_names = ['Model Name']
        if train_loader is not None:
            col_names.append('Train l/acc')
        if val_loader is not None:
            col_names.append('Val l/acc')
        if test_loader is not None:
            col_names.append('Test l/acc')
        model_summary_table = PrettyTable(col_names)
        model_list = []
        if test_original_model and not(self.retrain_masked):
            model_list = [('Original Model ' + self.model.name, self.model)]
        if test_masked_model:
            if self.model_masked is not None:
                model_list.append(
                    ('Masked Model ' + self.model_masked.name, self.model_masked))
        # saving heat map on some sample images
        if val_loader is not None and save_heat_map:
            for label in range(val_loader.dataset.n_classes):
                X, y = val_loader.dataset.sample_itm_class(label)
                for model in model_list:
                    if hasattr(model[-1], 'get_heat_map'):
                        # sample an image for class 0 for now
                        img_path = model[-1].get_heat_map(
                            X.clone(), y, self.storage_parent_dir, model[0] + '_' + mode_name)
                        self._logger.info(
                            'Created Heat map at {}'.format(img_path))
        results = []
        for model in model_list:
            current_model_results_dict = {}
            loss_acc = [0] * (len(col_names) - 1)
            start_indx = 0
            if train_loader is not None:
                loss_acc[start_indx] = test(
                    model[-1], train_loader, self.criterion)
                current_model_results_dict['loss_train'] = loss_acc[start_indx][0]
                current_model_results_dict['acc_train'] = loss_acc[start_indx][1]
                start_indx += 1
            if val_loader is not None:
                loss_acc[start_indx] = test(
                    model[-1], val_loader, self.criterion)
                start_indx += 1
            if test_loader is not None:
                loss_acc[start_indx] = test(
                    model[-1], test_loader, self.criterion)
                current_model_results_dict['loss_test'] = loss_acc[start_indx][0]
                current_model_results_dict['acc_test'] = loss_acc[start_indx][1]
                start_indx += 1
            resulting_info = [str(current_l_acc[0]) + ' / ' +
                              str(current_l_acc[1]) for current_l_acc in loss_acc]
            model_summary_table.add_row([model[0]]+resulting_info)
            results.append(current_model_results_dict)
        self._logger.info(str(model_summary_table))
        return results
