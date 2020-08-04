from .batchnorm import BatchNorm
import cvxpy as cp
import numpy as np
import torch
import copy


class BatchNorm2d(BatchNorm):
    """cvxpy layer of batchnorm 2d
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
        self.n_in_channels = pytorch_layer.num_features
        self.n_out_channels = self.n_in_channels
        self.layer_input = [
            cp.Variable((batch_size, input_size), name)
            for _ in range(pytorch_layer.num_features)
        ]
        if self.is_last_layer:
            self.last_layer_out = cp.Variable(
                (batch_size, pytorch_layer.num_features * input_size), name + "_last"
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
        constraints = []
        if self.activation is not None:
            constraints += self.activation.get_constraints(self, prev_layer)
        else:
            # for linear activations
            current_constraints = []
            for channel_indx in range(self.n_in_channels):
                upper_bound, _ = prev_layer.get_bounds(channel_indx)
                critical_prob = prev_layer.get_critical_neurons(channel_indx)
                if critical_prob is None:
                    keep_upper_bound = 0
                else:
                    keep_upper_bound = cp.multiply(1 - critical_prob, upper_bound)

                current_constraints += [
                    self.layer_input[channel_indx]
                    == prev_layer.get_computation_layer(channel_indx) - keep_upper_bound
                ]
            constraints += self.create_constraint(
                f"{self.name}_linear", current_constraints
            )
        if prev_layer.compute_critical_neurons:
            constraints += self.create_constraint(
                f"neuron_importance_bounds_{prev_layer.name}",
                [prev_layer.neuron_importance >= 0, prev_layer.neuron_importance <= 1],
            )
        return constraints

    def get_output_shape(self):
        """
        Returns:
            int -- size of the output of pytorch layer
        """
        return self.pytorch_layer.num_features * self.input_size

    def get_n_neurons(self):
        """
        Returns:
            int -- size of the output of pytorch layer
        """
        return self.pytorch_layer.num_features

    def get_cvxpy_variable(self, channel_indx=None):
        """get the cvxpy variable associated with this layer

        Returns:
            cvxpy.variable -- cvxpy variable holding output of current layer
        """
        if channel_indx is None:
            output_channels = cp.hstack(
                [
                    self.layer_input[cur_channel_indx]
                    for cur_channel_indx in range(self.n_in_channels)
                ]
            )
        else:
            output_channels = self.layer_input[channel_indx]
        return output_channels

    def get_computation_layer(self, channel_indx=0):
        """compute the output of this layer based on the weights biases and decision variable 

        Keyword Arguments:
            channel_indx {int} -- used to denote filter index but not used in 2d batchnorm setup  (default: {0})

        Returns:
            cvxpy.variable  -- returns a variable holding output of current layer after applying weights and biases
        """
        if channel_indx is None:
            return self._get_multi_channel_output_flat()
        normalized_batch = (
            self.layer_input[channel_indx] - self.running_mean[channel_indx]
        ) / (np.sqrt(self.running_var[channel_indx] + self.epsilon))
        if self.affine:
            return (normalized_batch * self.weights[channel_indx]) + self.bias[
                channel_indx
            ]
        return normalized_batch

    def _get_multi_channel_output_flat(self):
        output_channels = cp.hstack(
            [
                self.get_computation_layer(channel_indx)
                for channel_indx in range(self.n_in_channels)
            ]
        )
        return output_channels

    def get_bounds(self, channel_indx=None):
        """returns the bounds asssociated with input to this layer

        Returns:
            tuple -- upper and lower bound
        """
        if channel_indx is None:
            upper_bound = self.upper_bound.reshape(self.batch_size, -1)
            lower_bound = self.lower_bound.reshape(self.batch_size, -1)
        else:
            upper_bound = self.upper_bound[:, channel_indx, :].reshape(
                self.batch_size, -1
            )
            lower_bound = self.lower_bound[:, channel_indx, :].reshape(
                self.batch_size, -1
            )
        return upper_bound, lower_bound

    def _test(self):
        """routine used to test the current pooling implementation to make sure no discrepency between cvxpy and original pytorch layer
        """
        self.pytorch_layer.eval()
        pytorch_layer = copy.deepcopy(self.pytorch_layer).cpu()
        image_w_h = int(self.input_size ** 0.5)
        input_image = torch.rand(1, self.n_in_channels, image_w_h, image_w_h)
        output_tensor = pytorch_layer(input_image)[0]
        for channel in range(self.n_in_channels):
            current_channel = input_image[0, channel].squeeze().flatten().cpu().numpy()
            normalized_data = (current_channel - self.running_mean[channel]) / np.sqrt(
                self.running_var[channel] + self.epsilon
            )
            if self.affine:
                output_numpy = (self.weights[channel] * normalized_data) + self.bias[
                    channel
                ]
            else:
                output_numpy = normalized_data

            assert np.isclose(
                output_numpy,
                output_tensor[channel].detach().flatten().cpu().numpy(),
                atol=1e-6,
            ).all()

    def get_n_channels(self):
        """returns number of output channels

        Returns:
            int: number of output channels
        """        
        return self.n_out_channels
