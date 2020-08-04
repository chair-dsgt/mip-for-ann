import torch.nn as nn
import torch
import cvxpy as cp
import copy
import numpy as np
import os
from scipy.stats import norm

from layers_modules import layers_modules_maps, linear_layers_maps, ignored_layers_map
from activation_modules import activations_layer_map
from .cp_losses import softmax_loss
from training.utils import bound_propagation, test_batch, device
from contextlib import redirect_stdout
from training import Mode
import gc
from .sparsify_base import SparsifyBase


class SparsifyModel(SparsifyBase):
    def __init__(
        self,
        model_train_obj,
        sparsification_weight=5,
        threshold=1e-3,
        relaxed_constraints=False,
        mean_threshold=False,
    ):
        """initialization of sparsify model object

        Arguments:
            model_train_obj {ModelTrain} -- ModelTrain object used to train/test/fine tune the model

        Keyword Arguments:
            sparsification_weight {int} -- value of the \lambda used in the loss function (default: {5})
            threshold {float} -- the value of the cutting threshold to prune neurons any neuron having a score less than this one will be pruned (default: {1e-3})
            relaxed_constraints {bool} -- a flag used to relax ReLU constraints {0,1} to continuous range [0-1] (default: {False})
        """
        super().__init__(
            model_train_obj,
            sparsification_weight,
            threshold,
            relaxed_constraints,
            mean_threshold,
        )
        self.model_bounds = None
        self.model_constraints = []
        self.model_layers = []
        self.batch_size = None
        self.solved_mip = False

        # saved cvxpy model path
        self.model_name = self.model_train.get_model_to_sparsify().name
        self.save_dir = os.path.join(
            self.model_train.storage_parent_dir,
            "model_{}_cvxpy.pt".format(self.model_name),
        )
        # creating masked model
        self.model_train.swap_pytorch_layers()
        # saved importance score per layer after doing the sparsification
        self.neuron_importance_score = {}
        self.compressable_layers_indices = []

    def create_bounds(self, initial_bounds, init_layers=True):
        """create upper/lower bounds of the model

        Arguments:
            initial_bounds {np.array} -- upper and lower bound of input batch
        """
        model = self.model_train.get_model_to_sparsify()
        model.eval()
        self.batch_size = initial_bounds[0].shape[0]
        self.model_bounds = bound_propagation(model, initial_bounds)
        self._logger.info("Created Model {} Bounds".format(self.model_name))
        if (len(self.model_layers) == 0 and init_layers) or (
            len(self.model_layers) > 0
            and self.model_layers[0].batch_size != self.batch_size
        ):
            self._initialize_model_layers()
        self._logger.info("Created Model {} Layers".format(self.model_name))

    def sparsify_model(
        self,
        input_data_flatten,
        input_data_labels,
        mode=Mode.MASK,
        use_cached=True,
        start_pruning_from=None,
        save_neuron_importance=True,
    ):
        """computes the neuron importance using solver and sparsifies the model

        Arguments:
            input_data_flatten {np.array} -- batch of input data to the solver
            input_data_labels {list} -- labels of the input batch

        Keyword Arguments:
            mode {enum} -- masking mode used (default: {Mode.MASK})
            use_cached {bool} -- flag when enabled cached solver result from previous run will be used (default: {True})
            start_pruning_from {int} -- index of initial layer that will be represented in MIP and pruned from (default: {None})

        Returns:
            float -- percentage of parameters removed
        """
        debug_constraints = False  # flag to debug which constraints are causing error
        removed_params_percentage = 0
        if os.path.isfile(self.save_dir) and use_cached:
            self._load_cp_layers()
        elif not (self.solved_mip):
            prediction_results = test_batch(
                self.model_train.get_model_to_sparsify(),
                input_data_flatten,
                torch.from_numpy(input_data_labels).to(device),
            )
            self._logger.info(prediction_results.print_batch_probabilties())
            self.start_pruning_from = start_pruning_from
            self._create_constraints(input_data_flatten)
            prob = self._create_cp_loss(
                input_data_labels, debug_constraints=debug_constraints
            )
            if debug_constraints:
                self._debug_constraints(input_data_labels)
            else:
                # now run the solver on the input variables
                self._logger.info(
                    "[Exp] Getting Neuron Importance Score for model {} with sparsification score {}".format(
                        self.model_name, self.sparsification_weight
                    )
                )
                self._log_memory()
                objective_value = self._run_solver(prob)
                if prob.status in ["infeasible", "unbounded"]:
                    # Error occured and we need to debug the cvxpy model
                    self._logger.exception(
                        "Problem status {} , Now debugging infeasibility".format(
                            prob.status
                        )
                    )
                    self._debug_constraints(input_data_labels)
                    return 0
                solve_time = prob.solution.attr["solve_time"]
                self._logger.info(
                    "[solver] Solver Objective value {} in {} seconds".format(
                        objective_value, str(solve_time)
                    )
                )
                self.neuron_importance_score = {}
                for layer in self.model_layers[:-1]:
                    if layer.compute_critical_neurons:
                        self.neuron_importance_score[
                            layer.layer_indx
                        ] = layer.neuron_importance.value
                if save_neuron_importance:
                    self._save_cp_layers()
                self.solved_mip = True

        removed_params_percentage = self._filter_critical_neurons(mode)

        self._log_memory()
        return removed_params_percentage

    def _debug_constraints(self, input_data_labels):
        prob = self._create_cp_loss(input_data_labels, debug_constraints=True)
        for subprob in prob:
            self._logger.info("Added Constraint " + subprob[0])
            objective_value = self._run_solver(subprob[1])
            if subprob[1].status in ["infeasible", "unbounded"]:
                import pdb

                pdb.set_trace()
            self._logger.info("Objective Value " + str(objective_value))

    def _run_solver(self, prob):
        self._logger.debug("Now calling Cvxpy solver to solve")
        objective_value = None
        tmp_file_path = os.path.join(
            self.model_train.storage_parent_dir, "tmp_file.txt"
        )
        cp.settings.ERROR = [cp.settings.USER_LIMIT]
        cp.settings.SOLUTION_PRESENT = [
            cp.settings.OPTIMAL,
            cp.settings.OPTIMAL_INACCURATE,
            cp.settings.SOLVER_ERROR,
        ]
        with open(tmp_file_path, "w") as tmp_file:
            with redirect_stdout(tmp_file):
                objective_value = prob.solve(
                    verbose=self._logger.debug_param,
                    solver=cp.MOSEK,
                    mosek_params={"MSK_DPAR_OPTIMIZER_MAX_TIME": 4 * 60 * 60},
                )
        with open(tmp_file_path, "r") as tmp_file:
            solver_logs = tmp_file.read()
            self._logger.info("[solver] \n" + solver_logs)
        try:
            os.remove(tmp_file_path)
        except:
            pass
        return objective_value

    def _create_constraints(self, input_data_flatten):
        """creates the constraints associated with input model based on input data

        Arguments:
            input_data_flatten {np.array} -- input batch to the solver
        """
        self._logger.info("Started creating Cvxpy model constraints")
        self._propagate_layer_bounds()

        # Now getting model constraints
        self.model_constraints = []
        start_layer_indx = self.model_layers[0].layer_indx
        start_layer_indx = (
            start_layer_indx
            if self.start_pruning_from is None
            else self.start_pruning_from
        )
        for layer_indx, layer in enumerate(self.model_layers):
            if layer.layer_indx == start_layer_indx:
                for original_layer_indx in range(layer.layer_indx):
                    input_data_flatten = self.model_train.get_model_to_sparsify()[
                        original_layer_indx
                    ](input_data_flatten)
                input_data_flatten = input_data_flatten.detach().cpu().numpy()
                if type(layer) in linear_layers_maps:
                    input_data_flatten = input_data_flatten.reshape(
                        input_data_flatten.shape[0], -1
                    )
                else:
                    input_data_flatten = input_data_flatten.reshape(
                        input_data_flatten.shape[0], input_data_flatten.shape[1], -1
                    )
                self.model_constraints = layer.get_first_layer_constraints(
                    input_data_flatten
                )
                continue
            elif layer.layer_indx < start_layer_indx:
                continue
            self.model_constraints += layer.get_constraints(
                self.model_layers[layer_indx - 1]
            )

        # check constraints
        self._check_constraints()

    def _check_constraints(self):
        """checking if constraints are disciplined convex 
        """
        for const in self._get_constraints_list(self.model_constraints):
            if not (const.is_dcp()):
                self._logger.exception(
                    "Constraint not disciplined convex " + str(const)
                )

    def _initialize_model_layers(self):
        n_layers = len(self.model_train.get_model_to_sparsify())
        activation = None
        self.model_layers = []
        # creating cvxpy variables
        activation = None
        activation_name = "None"
        update_compressed_layers_indices = len(self.compressable_layers_indices) == 0
        for layer_indx, layer in enumerate(self.model_train.model_masked):
            if type(layer) in layers_modules_maps:
                if layer_indx == n_layers - 1:
                    is_last_layer = True
                else:
                    is_last_layer = False
                if hasattr(layer, "name"):
                    layer_name = layer.name
                else:
                    layer_name = str(layer) + str(layer_indx)
                if hasattr(layer, "input_size"):
                    layer_input_size = layer.input_size
                else:
                    layer_input_size = self.model_bounds[layer_indx - 1][0].shape
                    if len(layer_input_size) > 2:
                        layer_input_size = layer_input_size[-2] * layer_input_size[-1]
                    else:
                        layer_input_size = layer_input_size[1]
                current_layer_object = layers_modules_maps[type(layer)](
                    layer_name + "_" + str(layer_indx),
                    layer_indx,
                    layer,
                    self.batch_size,
                    input_size=layer_input_size,
                    activation=copy.deepcopy(activation),
                    is_last_layer=is_last_layer,
                    compute_critical_neurons=True,
                )
                if current_layer_object.compute_critical_neurons:
                    if update_compressed_layers_indices:
                        self.compressable_layers_indices.append(layer_indx)
                    if layer_indx in self.neuron_importance_score:
                        current_layer_object.neuron_importance.value = self.neuron_importance_score[
                            layer_indx
                        ]
                self.model_layers.append(current_layer_object)
                self._logger.debug(
                    "Created layer {} with activation {}".format(
                        current_layer_object.name, activation_name
                    )
                )
                activation = None
                activation_name = "None"
            elif type(layer) in ignored_layers_map:
                if type(layer) in activations_layer_map:
                    activation = activations_layer_map[type(layer)](
                        str(layer_indx + 1), relaxed_constraint=self.relaxed_constraints
                    )
                    activation_name = activation.name
                    self._logger.debug("Created activation {}".format(activation_name))
                continue
            else:
                self._logger.exception(
                    "This model layer is not supported " + str(type(layer))
                )

    def _propagate_layer_bounds(self):
        for layer in self.model_layers:
            layer.set_bounds(
                self.model_bounds[layer.layer_indx][0],
                self.model_bounds[layer.layer_indx][1],
            )
            # release memory used by np array if the solver was called before
            if (
                layer.compute_critical_neurons
                and layer.neuron_importance.value is not None
            ):
                layer.neuron_importance.value = None

            if (
                self.start_pruning_from is not None
                and layer.layer_indx < self.start_pruning_from
            ):
                layer.disable_compute_neurons()
        gc.collect()

    def _filter_critical_neurons(self, mode=Mode.MASK):
        """used to filter critical neurons based on solver's computed importance score

        Keyword Arguments:
            mode {enum} -- masking mode which can be random/mask/critical (default: {Mode.MASK})

        Returns:
            float -- percentage of parameters removed
        """
        self._logger.info("Started removing nodes with {}".format(mode.name))
        masked_indices_list = {}
        original_model_num_params = sum(
            p.numel() for p in self.model_train.get_model_to_sparsify().parameters()
        )
        sparsified_model_num_params = original_model_num_params
        sparsified_model_fc_num_params = original_model_num_params
        layer_ids = [layer.layer_indx for layer in self.model_layers[:-1]]
        for layer_indx, layer_id in enumerate(layer_ids):
            layer_threshold = self.threshold
            layer = self.model_layers[layer_indx]
            if (
                not (layer.compute_critical_neurons)
                or layer_id not in self.neuron_importance_score
            ):
                continue
            layer_neuron_importance = self.neuron_importance_score[layer_id]
            layer = self.model_layers[layer_indx]
            mean, std = norm.fit(layer_neuron_importance)
            max_score = np.max(layer_neuron_importance)
            min_score = np.min(layer_neuron_importance)
            neurons_shape = layer_neuron_importance.shape[0]
            if self.mean_threshold:
                layer_threshold = mean
            masked_indices = np.where(layer_neuron_importance < layer_threshold)[0]
            if mode == Mode.Random:
                # randomize masked indices from only critical neurons with same number of neurons as the non-critical ones
                important_neurons_indices = np.where(layer_neuron_importance > 0)[0]
                masked_indices = np.random.choice(
                    important_neurons_indices, size=masked_indices.shape
                )
            elif mode == Mode.CRITICAL:
                # Remove neurons having top score with same percentage as previously removed nodes
                if len(masked_indices) > 0:
                    masked_indices = layer_neuron_importance.argsort()[
                        -1 * len(masked_indices) :
                    ]
            else:
                # Mode Mask
                self._logger.info(
                    "[stats] Critical score of neurons  from layer {} having score {} +- {} [{} - {}] with threshold {}".format(
                        layer.name, mean, std, min_score, max_score, layer_threshold
                    )
                )
            sparsified_model_num_params -= layer.get_sparsified_param_size(
                masked_indices
            )
            if type(layer) in linear_layers_maps.values():
                sparsified_model_fc_num_params -= layer.get_sparsified_param_size(
                    masked_indices
                )
            perecentage_removed = len(masked_indices) * 100 / neurons_shape
            masked_indices_list[layer.layer_indx] = np.copy(masked_indices)
            self._logger.info(
                "Removed #{} neurons from layer {} with {}%".format(
                    str(len(masked_indices)), layer.name, str(perecentage_removed)
                )
            )
        parameters_removed_percentage = 100 - 100 * (
            sparsified_model_num_params / original_model_num_params
        )
        self._logger.info(
            "Original Model {} having # {}".format(
                self.model_train.model.name, str(original_model_num_params)
            )
        )
        self._logger.info(
            "Masked Model {} having # {} with mode {}".format(
                self.model_train.model_masked.name,
                str(sparsified_model_num_params),
                mode.name,
            )
        )
        self._logger.info(
            "[Results] Reduced Model {} size with {} %".format(
                self.model_train.model.name, str(parameters_removed_percentage)
            )
        )

        parameters_removed_percentage_fc = 100 - 100 * (
            sparsified_model_fc_num_params / original_model_num_params
        )
        self._logger.info(
            "[Results] Reduced Model {} 's fully connected params  size with {} %".format(
                self.model_train.model.name, str(parameters_removed_percentage_fc)
            )
        )
        self.model_train.set_mask_indices(
            masked_indices_list,
            suffix="_" + mode.name,
            save_masking_model=mode == Mode.MASK,
        )
        return parameters_removed_percentage

    def _create_cp_loss(self, input_data_labels, debug_constraints=False):
        """creates the solver's objective function sparsity + lambda softmax

        Arguments:
            input_data_labels {list} -- labels of the input batch

        Returns:
            cvxpy.Problem -- returns the cvxpy problem having the objective and the constraints
        """
        loss_softmax = (
            softmax_loss(self.model_layers[-1].get_layer_out(), input_data_labels)
            / self.batch_size
        )
        sum_critical_neurons = []
        n_neurons = 0
        for layer in self.model_layers[:-1]:
            if layer.compute_critical_neurons and layer.neuron_importance is not None:
                sum_critical_neurons.append(
                    cp.sum(layer.neuron_importance - 2) / layer.get_n_neurons()
                )
                n_neurons += layer.get_n_neurons()
        if n_neurons > 0:
            if len(sum_critical_neurons) == 1:
                sparsification_penalty = cp.sum(cp.vstack(sum_critical_neurons))
            else:
                sparsification_penalty = cp.sum_largest(
                    cp.vstack(sum_critical_neurons), int(len(sum_critical_neurons) - 1)
                )
            final_loss = cp.transforms.weighted_sum(
                objectives=[loss_softmax, sparsification_penalty],
                weights=np.array([self.sparsification_weight, 1]),
            )
        else:
            final_loss = loss_softmax
        if debug_constraints:
            # return a list of possible problems to know which constraint is the problem
            problems = []
            for constr_indx, constr in enumerate(self.model_constraints):
                problems.append(
                    (
                        constr.get_name(),
                        cp.Problem(
                            cp.Minimize(0),
                            self._get_constraints_list(
                                self.model_constraints[0 : (constr_indx + 1)]
                            ),
                        ),
                    )
                )
            return problems
        return cp.Problem(
            cp.Minimize(final_loss), self._get_constraints_list(self.model_constraints)
        )

    def _get_constraints_list(self, constraints):
        flatten = lambda l: [item for sublist in l for item in sublist]
        return flatten([constr.get_constraint() for constr in constraints])

    def _save_cp_layers(self):
        """save solver output including layer values, constraints and computed neuron importance score
        """
        try:
            if len(self.model_layers) == 0 or len(self.neuron_importance_score) == 0:
                self._logger.warning(
                    "CVXPY model {} layers saved are empty".format(self.model_name)
                )
            torch.save(
                {"neuron_importance": self.neuron_importance_score}, self.save_dir,
            )
            self._logger.info("saved model {} layers ".format(self.model_name))
        except Exception as e:
            self._logger.exception(str(e))

    def _load_cp_layers(self):
        """load the saved neuron importance score and cvxpy layers
        """
        try:
            checkpoint = torch.load(self.save_dir)
            self.neuron_importance_score = checkpoint["neuron_importance"]
            self._logger.info(
                "Loaded layers and constraints of model {}".format(self.model_name)
            )
        except Exception as e:
            self._logger.exception(str(e))

    def reset(self):
        """
        used to clean the memory used by the sparsification
        """
        self.model_constraints = []
        self.model_layers = []
        self.solved_mip = False
        gc.collect()


if __name__ == "__main__":

    # testing our approach
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    import torch.optim as optim
    from .model_train import ModelTrain
    from models import FullyConnectedBaselineModel

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mnist_train = datasets.MNIST(
        "../data", train=True, download=True, transform=transforms.ToTensor()
    )
    train_size = int(0.9 * len(mnist_train))
    val_size = len(mnist_train) - train_size
    mnist_train, mnist_val = torch.utils.data.random_split(
        mnist_train, [train_size, val_size]
    )

    mnist_test = datasets.MNIST(
        "../data", train=False, download=True, transform=transforms.ToTensor()
    )
    train_loader = DataLoader(mnist_train, batch_size=32, shuffle=True)
    val_loader = DataLoader(mnist_val, batch_size=100, shuffle=False)
    test_loader = DataLoader(mnist_test, batch_size=100, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    model_small = FullyConnectedBaselineModel()

    opt = optim.SGD(model_small.parameters(), lr=1e-1)
    model = ModelTrain(model_small, criterion, opt, "hamada", debug=True)
    model.train(train_loader, num_epochs=5)
    for X, y in val_loader:
        X, y = X.to(device), y.to(device)
        break

    sparsify = SparsifyModel(model)
    start_indx = 0
    end_indx = 10
    images = X[start_indx:end_indx]
    labels = y[start_indx:end_indx]
    initial_bound = ((images).clamp(min=0), (images).clamp(max=1))
    sparsify.create_bounds(initial_bound)
    sparsify.sparsify_model(
        images.flatten().cpu().numpy().reshape(images.shape[0], -1),
        labels.cpu().numpy(),
    )
    model.print_results(train_loader, val_loader, test_loader)
