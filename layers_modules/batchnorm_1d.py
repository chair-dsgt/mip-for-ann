from .batchnorm import BatchNorm
import cvxpy as cp
import numpy as np
import torch
import copy


class BatchNorm1d(BatchNorm):
    """cvxpy layer of batchnorm 1d
    """

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
        super().__init__(
            name,
            layer_indx,
            pytorch_layer,
            batch_size,
            input_size=input_size,
            activation=activation,
            is_last_layer=is_last_layer,
            compute_critical_neurons=False,
        )
        self.layer_input = cp.Variable((batch_size, pytorch_layer.num_features), name)
        if self.is_last_layer:
            self.last_layer_out = cp.Variable(
                (batch_size, pytorch_layer.num_features), name + "_last"
            )
        if self.testing_representation:
            self._test()

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

    def get_output_shape(self):
        """
        Returns:
            int -- size of the output of pytorch layer
        """
        return self.pytorch_layer.num_features

    def get_n_neurons(self):
        """
        Returns:
            int -- size of the output of pytorch layer
        """
        return self.pytorch_layer.num_features

    def get_cvxpy_variable(self, channel_indx=1):
        """get the cvxpy variable associated with this layer

        Returns:
            cvxpy.variable -- cvxpy variable holding output of current layer
        """
        return self.layer_input

    def get_computation_layer(self, channel_indx=0):
        """compute the output of this layer based on the weights biases and decision variable 

        Keyword Arguments:
            channel_indx {int} -- used to denote filter index but not used in 1d batchnorm setup  (default: {0})

        Returns:
            cvxpy.variable  -- returns a variable holding output of current layer after applying weights and biases
        """
        normalized_data = (self.layer_input - self.running_mean) / np.sqrt(
            self.running_var + self.epsilon
        )
        if self.affine:
            return cp.multiply(normalized_data, self.weights) + self.bias
        return normalized_data

    def _test(self):
        """routine used to test the current pooling implementation to make sure no discrepency between cvxpy and original pytorch layer
        """
        self.pytorch_layer.eval()
        pytorch_layer = copy.deepcopy(self.pytorch_layer).cpu()
        input_data = torch.rand(1, self.input_size)
        output_tensor = pytorch_layer(input_data)[0]
        input_data = input_data.squeeze().flatten().cpu().numpy()
        normalized_data = (input_data - self.running_mean) / np.sqrt(
            self.running_var + self.epsilon
        )
        if self.affine:
            output_numpy = (self.weights * normalized_data) + self.bias
        else:
            output_numpy = normalized_data

        assert np.isclose(
            output_numpy, output_tensor.detach().flatten().cpu().numpy(), atol=1e-6,
        ).all()

    def get_bounds(self, channel_indx=1):
        """returns the bounds asssociated with input to this layer

        Returns:
            tuple -- upper and lower bound
        """
        upper_bound = self.upper_bound
        lower_bound = self.lower_bound
        return upper_bound, lower_bound

    def get_n_channels(self):
        """return number of output channels
        """
        return 1

    def _reshape_input(self):
        if self.affine:
            self.weights = np.repeat(
                self.weights.reshape(1, -1), self.batch_size, axis=0
            )
            self.bias = np.repeat(self.bias.reshape(1, -1), self.batch_size, axis=0)
        self.running_mean = np.repeat(
            self.running_mean.reshape(1, -1), self.batch_size, axis=0
        )
        self.running_var = np.repeat(
            self.running_var.reshape(1, -1), self.batch_size, axis=0
        )
