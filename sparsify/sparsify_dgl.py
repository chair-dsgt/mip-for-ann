from .sparsify_model import SparsifyModel
from dataset import CustomLoss
import numpy as np
from training import Mode
import torch
import copy
import os
from training.utils import device
"""Decoupled layer wise sparsification 
"""

class SparsifyDgl:
    def __init__(
        self, sparsify_object, mip_data_loader, n_output_classes=1,
    ):
        self.mip_data_loader = mip_data_loader
        self.n_output_classes = n_output_classes
        self.sparsify_object = sparsify_object
        self.model_train = self.sparsify_object.model_train
        self._logger = self.model_train._logger

    def _sparsify(
        self,
        X,
        y,
        mode=Mode.MASK,
        test_original_model=True,
        test_masked_model=True,
        log_results=True,
        start_pruning_from=None,
        save_neuron_importance=True,
    ):
        """calls MIP solver on input batch of data points
        
        Arguments:
            X {np.array} -- input numpy array of data points
            y {np.array} -- list of labels associated with input data points
        
        Keyword Arguments:
            mode {Mode} -- sparsification mode (Mask, Critical, Random) (default: {Mode.MASK})
            test_original_model {bool} -- flag when enabled original model will be evaluated (default: {True})
            test_masked_model {bool} -- flag when enabled sparsified model will be evaluated(default: {True})
            log_results {bool} -- flag when enabled model evaluation results will be added to the log file (default: {True})
            start_pruning_from {int} -- index of initial layer that will be represented in MIP and pruned from (default: {None})

        Returns:
            tuple(PrettyTable, float, SparsifyModel) -- returns table of evaluated model results , percentage of parameters pruned and the Sparsification Object
        """

        self.sparsify_object.get_sparsify_object().solved_mip = False
        parameters_removed_percentage = self.sparsify_object.sparsify_model(
            X,
            y,
            mode=mode,
            use_cached=False,
            start_pruning_from=start_pruning_from,
            save_neuron_importance=save_neuron_importance,
        )
        return parameters_removed_percentage

    def sparsify_model(
        self, X, y, mode=Mode.MASK, use_cached=False, start_pruning_from=None,
    ):
        """sparsify model sequentially each class idndependently then we take the average
        
        Arguments:
            X {np.array} -- array of input batch of data points for the solver which are not used in the seq. run
            y {np.array} -- list of labels and can be safely None 
        
        Keyword Arguments:
            mode {Mode} -- mode of pruning Mask, Critical, Random (default: {Mode.MASK})
            use_cached {bool} -- bool flag to enable loading of previously cached models (default: {False})
        Returns:
            float -- percentage of removal of parameters
        """
        sparsify_base = self.sparsify_object.get_sparsify_object()
        if mode == Mode.MASK:
            neuron_importance_score = {}
            if os.path.isfile(sparsify_base.save_dir) and use_cached:
                sparsify_base._load_cp_layers()
                neuron_importance_score = sparsify_base.neuron_importance_score
            else:
                n_submodules = sparsify_base.model_train.len_dgl_submodules()
                initial_bounds = self.mip_data_loader.get_initial_bounds(X)
                for submodule_indx in range(n_submodules):
                    self._logger.info(
                        f"Started sparsifying submodule {submodule_indx} out of {n_submodules - 1}"
                    )
                    sparsify_base.model_train.enable_submodule_dgl(submodule_indx)

                    def custom_dgl_evaluator(model, input_image, targets):
                        # default evaluator based on weighted accuracyto get correctly predicted with lower probs. to allow sparsification
                        with torch.no_grad():
                            representation = sparsify_base.model_train.get_input_representation(
                                input_image
                            )
                            criterion = CustomLoss()
                            tmp_model = model.to(device)
                            logits = tmp_model(representation)
                            return criterion(logits, targets).item(), None

                    self.mip_data_loader.set_evaluator_function(custom_dgl_evaluator)
                    self.mip_data_loader.set_model(
                        sparsify_base.model_train.get_model_to_sparsify()
                    )
                    X, y, initial_bounds = next(self.mip_data_loader)
                    representation = sparsify_base.model_train.get_input_representation(
                        X
                    )
                    original_model_layers_indxs = (
                        sparsify_base.model_train.get_dgl_layers_to_sparsify()
                    )
                    if submodule_indx != 0:
                        sparsify_base.model_train.enable_submodule_dgl(-1)
                        sparsify_base.create_bounds(initial_bounds, init_layers=False)
                        original_model_bounds = sparsify_base.model_bounds
                        initial_bounds = original_model_bounds[
                            original_model_layers_indxs[0] - 1
                        ]
                        initial_bounds = [
                            torch.from_numpy(initial_bounds[0]).type_as(X),
                            torch.from_numpy(initial_bounds[1]).type_as(X),
                        ]
                        sparsify_base.model_train.enable_submodule_dgl(submodule_indx)
                    else:
                        initial_bounds = self.mip_data_loader.get_initial_bounds(
                            representation
                        )

                    sparsify_base.model_train.swap_pytorch_layers()
                    sparsify_base.create_bounds(initial_bounds)
                    sparsify_base.neuron_importance_score = {}
                    sparsify_base._initialize_model_layers()
                    parameters_removed_percentage = self._sparsify(
                        representation,
                        y,
                        test_original_model=False,
                        test_masked_model=False,
                        log_results=False,
                        save_neuron_importance=False
                    )
                    if submodule_indx == n_submodules - 1:
                        for layer_indx in sparsify_base.neuron_importance_score:
                            neuron_importance_score[
                                layer_indx + original_model_layers_indxs[0]
                            ] = sparsify_base.neuron_importance_score[layer_indx]
                    else:
                        for layer_indx in original_model_layers_indxs:
                            equivalent_index = (
                                layer_indx - original_model_layers_indxs[0]
                            )
                            if (
                                equivalent_index
                                in sparsify_base.neuron_importance_score
                            ):
                                neuron_importance_score[
                                    layer_indx
                                ] = sparsify_base.neuron_importance_score[
                                    equivalent_index
                                ]
                sparsify_base.model_bounds = original_model_bounds
            sparsify_base.model_train.enable_submodule_dgl(-1)
            sparsify_base.model_train.swap_pytorch_layers()
            sparsify_base.neuron_importance_score = {}
            initial_bounds = self.mip_data_loader.get_initial_bounds(X)
            sparsify_base.create_bounds(initial_bounds)
            sparsify_base._initialize_model_layers()
            sparsify_base.neuron_importance_score = neuron_importance_score

            for layer in sparsify_base.model_layers:
                layer.enable_compute_neurons()
            parameters_removed_percentage = sparsify_base._filter_critical_neurons()
            sparsify_base.solved_mip = True
            sparsify_base._save_cp_layers()
        else:
            parameters_removed_percentage = sparsify_base.sparsify_model(
                None, None, mode=mode, use_cached=False,
            )
        return parameters_removed_percentage

    def reset(self):
        self.sparsify_object.reset()
