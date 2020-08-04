import cvxpy as cp
import numpy as np
from layers_modules.submodule import SubModule


class FullyConnected(SubModule):
    def __init__(
        self,
        name,
        layer_indx,
        pytorch_layer,
        batch_size,
        input_size=None,
        activation=None,
        is_last_layer=False,
        compute_critical_neurons=False,
    ):
        """Initialization of fully connected layer including cvxpy solver variables and constraints

        Arguments:
            SubModule {[type]} -- [description]
            name {string} -- identifier of the layer
            layer_indx {int} -- layer order index in the pytorch model
            pytorch_layer {nn.module} -- pytorch layer that we are representing to represent in cvxpy
            batch_size {int} -- size of the input batch to the solver

        Keyword Arguments:
            input_size {int} -- size of the input  (default: {None})
            activation {activation} -- object holding activation applied to this layer (default: {None})
            is_last_layer {bool} -- a flag to denote if this is output layer or not (default: {False})
        """
        super().__init__(
            name,
            layer_indx,
            pytorch_layer,
            batch_size,
            input_size=input_size,
            activation=activation,
            is_last_layer=is_last_layer,
            compute_critical_neurons=compute_critical_neurons,
        )
        self.weights = None
        self.bias = None
        self._extract_pt_params()
        # setting cvxpy variable
        self.last_layer_out = None
        self.layer_input = cp.Variable((batch_size, pytorch_layer.in_features), name)
        self.neuron_importance = None
        if self.is_last_layer:
            self.last_layer_out = cp.Variable(
                (batch_size, pytorch_layer.out_features), f"{name}_last"
            )
            self.compute_critical_neurons = False
        elif self.compute_critical_neurons:
            self.neuron_importance = cp.Variable(
                (pytorch_layer.out_features), f"{name}_neuron_importance"
            )

    def _extract_pt_params(self):
        self.weights = self.pytorch_layer.linear.weight.detach().cpu().numpy().T
        bias = self.pytorch_layer.linear.bias.detach().cpu().numpy()
        self.bias = np.repeat(bias.reshape(1, -1), self.batch_size, axis=0)

    def get_constraints(self, prev_layer):
        """get constraints of the current layer

        Arguments:
            prev_layer {layers_modules.} -- previous layer object holding cvxpy variables

        Returns:
            list -- list of constraints used by the solver
        """
        prev_layer_computation = prev_layer.get_computation_layer(None)
        constraints = []
        if self.activation is None:
            critical_prob = prev_layer.get_critical_neurons(None)
            upper_bound, _ = prev_layer.get_bounds()
            if critical_prob is None:
                keep_upper_bound = 0
            else:
                keep_upper_bound = cp.multiply(1 - critical_prob, upper_bound)
            constraints += self.create_constraint(
                f"{self.name}_linear_eq",
                [self.layer_input == prev_layer_computation - keep_upper_bound],
            )
        else:
            constraints += self.activation.get_constraints(self, prev_layer)
        if prev_layer.compute_critical_neurons:
            constraints += self.create_constraint(
                f"neuron_importance_bounds_{prev_layer.name}",
                [prev_layer.neuron_importance >= 0, prev_layer.neuron_importance <= 1],
            )

        if self.is_last_layer:
            constraints += self.create_constraint(
                f"{self.name}_last_layer_eq",
                [self.last_layer_out == self.get_computation_layer()],
            )
        return constraints

    def get_cvxpy_variable(self, channel_indx=None):
        """get the cvxpy variable associated with this layer

        Returns:
            cvxpy.variable -- cvxpy variable holding output of current layer
        """
        return self.layer_input

    def get_output_shape(self):
        """
        Returns:
            int -- size of the output of pytorch layer
        """
        return self.pytorch_layer.out_features

    def get_n_neurons(self):
        """
        Returns:
            int -- size of the output of pytorch layer
        """
        return self.pytorch_layer.out_features

    def get_critical_neurons(self, channel_indx=None):
        """returns the critical neurons decision variable used by the solver

        Keyword Arguments:
            channel_indx {int} -- used to denote filter index but not used in fully connected setup (default: {0})

        Returns:
            cvxpy.variable -- reshaped variable of neuron importance repeated for every batch item in the input to the solver
        """
        if not (self.compute_critical_neurons):
            return None
        return cp.vstack(
            [
                cp.reshape(self.neuron_importance, (1, self.neuron_importance.shape[0]))
                for _ in range(self.batch_size)
            ]
        )

    def get_computation_layer(self, channel_indx=None):
        """compute the output of this layer based on the weights biases and decision variable 

        Keyword Arguments:
            channel_indx {int} -- used to denote filter index but not used in fully connected setup  (default: {0})

        Returns:
            cvxpy.variable  -- returns a variable holding output of current layer after applying weights and biases
        """
        return self.layer_input @ self.weights + self.bias

    def get_first_layer_constraints(self, input_data):
        """returns constraints associated with first layer

        Arguments:
            input_data {np.array} -- the input batch to the current layer

        Returns:
            list -- constraint used by the solver
        """
        if input_data.ndim > 2:
            input_data = input_data.reshape(self.batch_size, -1)
        return self.create_constraint(
            f"{self.name}_input_equality", [self.layer_input == input_data]
        )

    def get_sparsified_param_size(self, masked_indices):
        """compute the number of pruned parameters for this layer

        Arguments:
            masked_indices {list} -- list of masked indices

        Returns:
            int -- number of pruned parameters
        """
        return self.pytorch_layer.get_sparsified_param_size(masked_indices)

    def get_bounds(self, channel_indx=None):
        """returns the bounds asssociated with input to this layer

        Returns:
            tuple -- upper and lower bound
        """
        upper_bound = self.upper_bound
        lower_bound = self.lower_bound
        return upper_bound, lower_bound

    def get_n_channels(self):
        """returns number of output channels
        """        
        return 1
