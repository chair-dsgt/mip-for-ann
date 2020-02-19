import cvxpy as cp
import numpy as np
from layers_modules.submodule import SubModule


class FullyConnected(SubModule):
    def __init__(self, name, layer_indx, batch_size, pytorch_layer, input_size=None, activation=None, is_last_layer=False):
        """Initialization of fully connected layer including cvxpy solver variables and constraints

        Arguments:
            SubModule {[type]} -- [description]
            name {string} -- identifier of the layer
            layer_indx {int} -- layer order index in the pytorch model
            batch_size {int} -- size of the input batch to the solver
            pytorch_layer {nn.module} -- pytorch layer that we are representing to represent in cvxpy

        Keyword Arguments:
            input_size {int} -- size of the input  (default: {None})
            activation {activation} -- object holding activation applied to this layer (default: {None})
            is_last_layer {bool} -- a flag to denote if this is output layer or not (default: {False})
        """
        super().__init__(name, layer_indx, pytorch_layer,
                         batch_size, input_size, activation, is_last_layer)
        self.weights = pytorch_layer.linear.weight.detach().cpu().numpy()
        self.bias = pytorch_layer.linear.bias.detach().cpu().numpy()
        # setting cvxpy variable
        self.neuron_importance = None
        self.last_layer_out = None
        self.bias_value = np.repeat(
            self.bias.reshape(1, -1), self.batch_size, axis=0)

        self.layer_out = cp.Variable(
            (batch_size, pytorch_layer.in_features), name)

        if self.is_last_layer:
            self.last_layer_out = cp.Variable(
                (batch_size, pytorch_layer.out_features), name+'_last')
        else:
            self.neuron_importance = cp.Variable(
                (pytorch_layer.out_features), name+'_neuron_importance')
        self.weights_value = self.weights.T

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
            upper_bound = prev_layer.upper_bound.reshape(self.batch_size, -1)
            lower_bound = prev_layer.lower_bound.reshape(self.batch_size, -1)
            if critical_prob is None:
                keep_lower_bound = 0
            else:
                keep_lower_bound = cp.multiply(1 - critical_prob, lower_bound)
            constraints += [
                self.layer_out >= prev_layer_computation -
                cp.multiply(1 - critical_prob, upper_bound),
                prev_layer_computation -
                cp.multiply(1 - critical_prob,
                            upper_bound) >= self.layer_out + keep_lower_bound,
            ]
        else:
            constraints += self.activation.get_constraints(self, prev_layer)
        if prev_layer.neuron_importance is not None:
            constraints += [
                prev_layer.neuron_importance >= 0,
                prev_layer.neuron_importance <= 1
            ]

        if self.is_last_layer:
            constraints += [self.last_layer_out ==
                            self.get_computation_layer()]
        return constraints

    def get_cvxpy_variable(self):
        """get the cvxpy variable associated with this layer

        Returns:
            cvxpy.variable -- cvxpy variable holding output of current layer
        """
        return self.layer_out

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

    def get_critical_neurons(self, channel_indx=0):
        """returns the critical neurons decision variable used by the solver
        
        Keyword Arguments:
            channel_indx {int} -- used to denote filter index but not used in fully connected setup (default: {0})
        
        Returns:
            cvxpy.variable -- reshaped variable of neuron importance repeated for every batch item in the input to the solver
        """        
        if not(self.compute_critical_neurons):
            return None
        return cp.vstack([cp.reshape(self.neuron_importance, (1, self.neuron_importance.shape[0])) for _ in range(self.batch_size)])

    def get_computation_layer(self, channel_indx=0):
        """compute the output of this layer based on the weights biases and decision variable 
        
        Keyword Arguments:
            channel_indx {int} -- used to denote filter index but not used in fully connected setup  (default: {0})
        
        Returns:
            cvxpy.variable  -- returns a variable holding output of current layer after applying weights and biases
        """        
        return self.layer_out @ self.weights_value + self.bias_value

    def get_first_layer_constraints(self, input_data):
        """returns constraints associated with first layer
        
        Arguments:
            input_data {np.array} -- the input batch to the current layer
        
        Returns:
            list -- constraint used by the solver
        """        
        if input_data.ndim > 2:
            input_data = input_data.reshape(self.batch_size, -1)
        return [self.layer_out == input_data]

    def get_sparsified_param_size(self, masked_indices):
        """compute the number of pruned parameters for this layer
        
        Arguments:
            masked_indices {list} -- list of masked indices
        
        Returns:
            int -- number of pruned parameters
        """        
        return len(masked_indices) * self.pytorch_layer.in_features

    def get_bounds(self):
        """returns the bounds asssociated with input to this layer
        
        Returns:
            tuple -- upper and lower bound
        """        
        upper_bound = self.upper_bound
        lower_bound = self.lower_bound
        return upper_bound, lower_bound
