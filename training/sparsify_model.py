import torch.nn as nn
import torch
import torch
import cvxpy as cp
import copy
import numpy as np
from prettytable import PrettyTable
import os
from scipy.stats import norm

from custom_pt_layers import MaskedLinear, Flatten
from layers_modules import FullyConnected, layers_modules_maps, conv_layers, linear_layers_maps, ignored_layers_map
from activation_modules import ReLUActivation, activations_layer_map
from .cp_losses import softmax_loss
from .utils import test, bound_propagation, test_batch, device
from .logger import Logger
from .utils import Mode


class SparsifyModel:
    def __init__(self, model_train_obj, sparsification_weight=5, threshold=1e-3, relaxed_constraints=False):
        """initialization of sparsify model object

        Arguments:
            model_train_obj {ModelTrain} -- ModelTrain object used to train/test/fine tune the model

        Keyword Arguments:
            sparsification_weight {int} -- value of the lambda used in the loss function (default: {5})
            threshold {float} -- the value of the cutting threshold to prune neurons any neuron having a score less than this one will be pruned (default: {1e-3})
            relaxed_constraints {bool} -- a flag used to relax ReLU constraints {0,1} to continuous range [0-1] (default: {False})
        """
        self.model_train = model_train_obj
        self.model_bounds = None
        self.model_constraints = []
        self.model_layers = []
        self.batch_size = None
        self.threshold = threshold
        self.sparsification_weight = sparsification_weight
        self._logger = model_train_obj._logger
        self.solved_mip = False

        # saved cvxpy model path
        self.model_name = self.model_train.model.name
        self.save_dir = os.path.join(
            self.model_train.storage_parent_dir, 'model_{}_cvxpy.pt'.format(self.model_name))
        # creating masked model
        self.model_train.swap_pytorch_layers()
        # used to relax relu
        self.relaxed_constraints = relaxed_constraints

    def create_bounds(self, initial_bounds):
        """create upper/lower bounds of the model

        Arguments:
            initial_bounds {np.array} -- upper and lower bound of input batch
        """
        self.model_train.model_masked.eval()
        self.model_train.model.eval()
        self.batch_size = initial_bounds[0].shape[0]
        self.model_bounds = bound_propagation(
            self.model_train.model, initial_bounds)
        self._logger.info('Created Model {} Bounds'.format(
            self.model_name))

    def sparsify_model(self, input_data_flatten, input_data_labels, mode=Mode.MASK, use_cached=True):
        """computes the neuron importance using solver and sparsifies the model

        Arguments:
            input_data_flatten {np.array} -- batch of input data to the solver
            input_data_labels {list} -- labels of the input batch

        Keyword Arguments:
            mode {enum} -- masking mode used (default: {Mode.MASK})
            use_cached {bool} -- flag when enabled cached solver result from previous run will be used (default: {True})

        Returns:
            float -- percentage of parameters removed
        """
        if os.path.isfile(self.save_dir) and use_cached:
            self._load_cp_layers()
        elif not(self.solved_mip):
            n_correct_itms, _ = test_batch(
                self.model_train.model, input_data_flatten, torch.from_numpy(input_data_labels).to(device))
            accuracy_mip_input = n_correct_itms * 100 / len(input_data_labels)
            self._logger.info(
                'Accuracy of Batch of data points input to the MIP #{}%'.format(accuracy_mip_input))
            self._create_constraints(input_data_flatten)
            prob = self._create_cp_loss(input_data_labels)
            # now run the solver on the input variables
            self._logger.info(
                'Getting Neuron Importance Score for model {} with sparsification score {}'.format(self.model_name, self.sparsification_weight))
            # solver can be changed to Gurobi Mosek or any free alternative but free solvers takes longer to sovle same problems
            objective_value = prob.solve(
                verbose=self._logger.debug_param, solver=cp.MOSEK, mosek_params={'MSK_IPAR_LOG_FILE': 1})
            solve_time = prob.solution.attr['solve_time']
            self._logger.info(
                'Solver Objective value {} in {} seconds'.format(
                    objective_value, str(solve_time))
            )
            self._save_cp_layers()
            self.solved_mip = True
        return self._filter_critical_neurons(mode)

    def _create_constraints(self, input_data_flatten):
        """creates the constraints associated with input model based on input data

        Arguments:
            input_data_flatten {np.array} -- input batch to the solver
        """
        self._logger.info('Started creating Cvxpy model constraints')
        n_layers = len(self.model_train.model)
        activation = None
        self.model_layers = []
        # creating cvxpy variables
        is_prev_layer_activation = False
        activation = None
        for layer_indx, layer in enumerate(self.model_train.model_masked):
            if type(layer) in layers_modules_maps:
                if layer_indx == n_layers - 1:
                    is_last_layer = True
                else:
                    is_last_layer = False

                current_layer_object = layers_modules_maps[type(layer)](
                    layer.name+'_'+str(layer_indx), layer_indx, self.batch_size, layer, input_size=layer.input_size, activation=copy.deepcopy(activation), is_last_layer=is_last_layer)
                current_layer_object.set_bounds(
                    self.model_bounds[layer_indx][0], self.model_bounds[layer_indx][1])
                self.model_layers.append(current_layer_object)
                if is_prev_layer_activation:
                    activation = None
                    is_prev_layer_activation = False
            elif type(layer) in ignored_layers_map:
                if type(layer) in activations_layer_map:
                    activation = activations_layer_map[type(layer)](
                        str(layer_indx+1), relaxed_constraint=self.relaxed_constraints)
                    is_prev_layer_activation = True
                continue
            else:
                self._logger.exception(
                    'This model layer is not supported ' + str(type(layer)))

        # Now getting model constraints
        self.model_constraints = []
        for layer_indx, layer in enumerate(self.model_layers):
            if layer_indx == 0:
                for original_layer_indx in range(layer.layer_indx):
                    input_data_flatten = self.model_train.model[original_layer_indx](
                        input_data_flatten)
                input_data_flatten = input_data_flatten.detach().cpu().numpy()
                if type(layer) in linear_layers_maps:
                    input_data_flatten = input_data_flatten.reshape(
                        X.shape[0], -1)
                self.model_constraints = layer.get_first_layer_constraints(
                    input_data_flatten)
                continue
            self.model_constraints += layer.get_constraints(
                self.model_layers[layer_indx-1])

        # check constraints
        self._check_constraints()

    def _check_constraints(self):
        """checking if constraints are disciplined convex 
        """
        for const in self.model_constraints:
            if not(const.is_dcp()):
                self._logger.exception(
                    'Constraint not disciplined convex ' + str(const))

    def _filter_critical_neurons(self, mode=Mode.MASK):
        """used to filter critical neurons based on solver's computed importance score

        Keyword Arguments:
            mode {enum} -- masking mode which can be random/mask/critical (default: {Mode.MASK})

        Returns:
            float -- percentage of parameters removed
        """
        self._logger.info('Started removing nodes with {}'.format(mode.name))
        masked_indices_list = []
        original_model_num_params = sum(
            p.numel() for p in self.model_train.model.parameters())
        sparsified_model_num_params = original_model_num_params
        total_fully_connected_neurons = 0
        pruned_neurons_fully_connected = 0
        for layer in self.model_layers[:-1]:
            mean, std = norm.fit(layer.neuron_importance.value)
            max_score = np.max(layer.neuron_importance.value)
            min_score = np.min(layer.neuron_importance.value)
            layer_threshold = self.threshold
            if not(layer.compute_critical_neurons):
                masked_indices = []
                neurons_shape = 1
            else:
                neurons_shape = layer.neuron_importance.shape[0]
                masked_indices = np.where(
                    layer.neuron_importance.value < layer_threshold)[0]
                if mode == Mode.Random:
                    # randomize masked indices from only critical neurons with same number of neurons as the non-critical ones
                    important_neurons_indices = np.where(
                        layer.neuron_importance.value > np.mean(layer.neuron_importance.value))[0]
                    masked_indices = np.random.choice(
                        important_neurons_indices, size=masked_indices.shape)
                elif mode == Mode.CRITICAL:
                    # Remove neurons having top score with same percentage as previously removed nodes
                    if len(masked_indices) > 0:
                        masked_indices = layer.neuron_importance.value.argsort(
                        )[-1 * len(masked_indices):]
                else:
                    # Mode Mask
                    self._logger.info('Critical score of neurons  from layer {} having score {} +- {} [{} - {}] with threshold {}'.format(
                        layer.name, mean, std, min_score, max_score, layer_threshold))
            sparsified_model_num_params -= layer.get_sparsified_param_size(
                masked_indices)
            total_fully_connected_neurons += neurons_shape
            pruned_neurons_fully_connected += len(masked_indices)
            perecentage_removed = len(
                masked_indices)*100 / neurons_shape
            masked_indices_list.append(
                (layer.layer_indx, np.copy(masked_indices)))
            self._logger.info('Removed #{} neurons from layer {} with {}%'.format(
                str(len(masked_indices)), layer.name, str(perecentage_removed)))
        parameters_removed_percentage = 100 - 100 * \
            (sparsified_model_num_params / original_model_num_params)
        self._logger.info('Original Model {} having # {}'.format(
            self.model_train.model.name, str(original_model_num_params)))
        self._logger.info('Masked Model {} having # {} with mode {}'.format(
            self.model_train.model.name, str(sparsified_model_num_params), mode.name))
        self._logger.info('Reduced Model {} size with {} %'.format(
            self.model_train.model.name, str(parameters_removed_percentage)
        ))
        parameters_removed_percentage = 100 * \
            (pruned_neurons_fully_connected / total_fully_connected_neurons)
        self._logger.info('Reduced Model {} \'s fully connected params  size with {} %'.format(
            self.model_train.model.name, str(parameters_removed_percentage)
        ))
        self.model_train.set_mask_indices(
            masked_indices_list, suffix='_'+mode.name, save_masking_model=mode == Mode.MASK)
        return parameters_removed_percentage

    def _create_cp_loss(self, input_data_labels):
        """creates the solver's objective function sparsity + lambda softmax

        Arguments:
            input_data_labels {list} -- labels of the input batch

        Returns:
            cvxpy.Problem -- returns the cvxpy problem having the objective and the constraints
        """
        loss_softmax = softmax_loss(
            self.model_layers[-1].get_layer_out(), input_data_labels) / self.batch_size
        sum_critical_neurons = []
        n_neurons = 0
        for layer in self.model_layers[:-1]:
            if layer.compute_critical_neurons:
                sum_critical_neurons.append(
                    cp.sum(layer.neuron_importance - 2))
                n_neurons += layer.get_n_neurons()
        sparsification_penalty = cp.sum_largest(
            cp.vstack(sum_critical_neurons), int(len(self.model_layers[:-1]) - 1)) / n_neurons
        final_loss = cp.transforms.weighted_sum(objectives=[loss_softmax, sparsification_penalty], weights=np.array([
            self.sparsification_weight,  1]))
        return cp.Problem(cp.Minimize(final_loss), self.model_constraints)

    def _save_cp_layers(self):
        """save solver output including layer values, constraints and computed neuron importance score
        """
        try:
            if len(self.model_layers) == 0:
                self._logger.warning(
                    'CVXPY model {} layers saved are empty'.format(self.model_name))
            torch.save({
                'constraints': self.model_constraints,
                'layers': self.model_layers
            }, self.save_dir)
            self._logger.info('saved model {} layers '.format(self.model_name))
        except Exception as e:
            self._logger.exception(str(e))

    def _load_cp_layers(self):
        """load the saved neuron importance score and cvxpy layers
        """
        try:
            checkpoint = torch.load(self.save_dir)
            self.model_constraints = checkpoint['constraints']
            self.model_layers = checkpoint['layers']
            self._logger.info(
                'Loaded layers and constraints of model {}'.format(self.model_name))
        except Exception as e:
            self._logger.exception(str(e))


if __name__ == "__main__":

    # testing our approach
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    import torch.optim as optim
    from .model_train import ModelTrain
    from models import FullyConnectedBaselineModel
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mnist_train = datasets.MNIST(
        "../data", train=True, download=True, transform=transforms.ToTensor())
    train_size = int(0.9 * len(mnist_train))
    val_size = len(mnist_train) - train_size
    mnist_train, mnist_val = torch.utils.data.random_split(
        mnist_train, [train_size, val_size])

    mnist_test = datasets.MNIST(
        "../data", train=False, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(mnist_train, batch_size=32, shuffle=True)
    val_loader = DataLoader(mnist_val, batch_size=100, shuffle=False)
    test_loader = DataLoader(mnist_test, batch_size=100, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    model_small = FullyConnectedBaselineModel()

    opt = optim.SGD(model_small.parameters(), lr=1e-1)
    model = ModelTrain(model_small, criterion, opt, 'hamada', debug=True)
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
    sparsify.sparsify_model(images.flatten().cpu().numpy().reshape(
        images.shape[0], -1), labels.cpu().numpy())
    model.print_results(train_loader, val_loader, test_loader)
