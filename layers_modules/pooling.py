from .submodule import SubModule
import numpy as np
import cvxpy as cp
from training.utils import square_indx_to_flat


class Pooling(SubModule):
    """Parent class of all pooling layers
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
        compute_critical_neurons=True,
    ):
        """initializing pooling parent class
        
        Arguments:
            name {string} -- name of the current layer
            layer_indx {int} -- index of the current layer in the model that will be sparsified
            pytorch_layer {nn.module} -- pytorch layer 
            batch_size {int} -- size of the batch of data points fed to the MIP
        
        Keyword Arguments:
            input_size {tuple(int, int)} -- size of the input image to pooling (default: {None})
            activation {Activation} -- activation applied to this layer eg. relu and None if no activation(default: {None})
            is_last_layer {bool} -- flag when enabled denotes that this is the last layer (default: {False})
            compute_critical_neurons {bool} -- flag to compyte neuron importance score (default: {False})
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
        self.kernel_size = None
        self.stride_size = None
        self.prev_layer = None
        self.output_size = [None, None]
        self.image_w_h = int(self.input_size ** 0.5)  # for symmetric images
        self.n_out_channels = None
        self.compute_critical_neurons = False

    def get_constraints(self, prev_layer):
        """returns constraints of the current pooling layer
        
        Arguments:
            prev_layer {SubModule} -- previous layer cvxpy
        
        Returns:
            list(constraint) -- returns list of constraints associated with this pooling layer
        """
        self._set_prev_conv(prev_layer)
        self.layer_input = [
            cp.Variable(
                (self.batch_size, prev_layer.get_output_shape() // self.n_out_channels),
                f"{self.name}_{ch_indx}",
            )
            for ch_indx in range(self.n_out_channels)
        ]
        if self.testing_representation:
            self._test()  # testing the current pooling layer to avoid any bugs
        constraints = self._init_constraints(prev_layer)
        if self.activation is not None:
            constraints += self.activation.get_constraints(self, prev_layer)
        else:
            # for linear activations
            for channel_indx in range(self.n_out_channels):
                upper_bound, _ = prev_layer.get_bounds(channel_indx)
                critical_prob = prev_layer.get_critical_neurons(channel_indx)
                if critical_prob is None:
                    keep_upper_bound = 0
                else:
                    keep_upper_bound = cp.multiply(1 - critical_prob, upper_bound)

                constraints += self.create_constraint(
                    f"{self.name}_linear_{channel_indx}",
                    [
                        self.layer_input[channel_indx]
                        == self.prev_layer.get_computation_layer(channel_indx)
                        - keep_upper_bound
                    ],
                )

        if prev_layer.compute_critical_neurons:
            constraints += self.create_constraint(
                f"neuron_importance_bounds_{prev_layer.name}",
                [prev_layer.neuron_importance >= 0, prev_layer.neuron_importance <= 1],
            )
        return constraints

    def get_cvxpy_variable(self, channel_indx=None):
        """get the cvxpy variable associated with this layer

        Returns:
            cvxpy.variable -- cvxpy variable holding output of current layer
        """
        if channel_indx is None:
            output_channels = cp.hstack(
                [
                    self.layer_input[cur_channel_indx]
                    for cur_channel_indx in range(self.n_out_channels)
                ]
            )
        else:
            output_channels = self.layer_input[channel_indx]
        return output_channels

    def get_computation_layer(self, channel_idx=None):
        """returns pooling computation score applied at specified channel index
        
        Keyword Arguments:
            channel_indx {int} -- channel index at which the conv. operation will be applied if none returns concatenation of all channels (default: {0})
        
        Returns:
            cvxpy.variable -- output computation of current convolution layer
        """
        if channel_idx is None:
            return self._get_multi_channel_output_flat()
        return self._get_computation(self.layer_input, channel_idx)

    def _get_multi_channel_output_flat(self):
        """used to stack output of multiple channels into one
        
        Returns:
            cvxpy.variable -- variable having stacked version of all channels output from current layer
        """
        output_channels = cp.hstack(
            [
                self.get_computation_layer(channel_indx)
                for channel_indx in range(self.n_out_channels)
            ]
        )
        return output_channels

    def _update_pooling_params(self):
        """updates pooling params based on the pytorch layer
        """
        self.kernel_size = self.pytorch_layer.kernel_size
        self.stride_size = self.pytorch_layer.stride
        if isinstance(self.stride_size, int):
            self.stride_size = [self.stride_size for _ in range(2)]
        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size for _ in range(2)]
        self.output_size[0] = int(
            np.floor((self.image_w_h - self.kernel_size[1]) / self.stride_size[1]) + 1
        )
        self.output_size[1] = int(
            np.floor((self.image_w_h - self.kernel_size[0]) / self.stride_size[0]) + 1
        )

    def _get_pool_indices(self):
        """returns indices of pixles that will be pooled together to redice size of the image 
        
        Returns:
            list(list) -- list of all indices each item containing a list of indices that will be pooled together at that output pixel
        """
        indices = []
        for i in range(self.output_size[1]):
            for j in range(self.output_size[0]):
                current_block_indices = []
                for row_indx in range(self.kernel_size[0]):
                    for col_indx in range(self.kernel_size[1]):
                        current_block_indices.append(
                            square_indx_to_flat(
                                i * self.stride_size[0] + row_indx,
                                j * self.stride_size[1] + col_indx,
                                self.image_w_h,
                            )
                        )
                indices.append(current_block_indices)
        return indices

    def _set_prev_conv(self, prev_layer):
        """sets the previous cvxpy layer
        
        Arguments:
            prev_layer {SubModule} -- previous cvxpy layer
        """
        self.prev_layer = prev_layer
        self.n_out_channels = prev_layer.n_out_channels

    def get_n_neurons(self):
        return 0

    def get_sparsified_param_size(self, masked_indices):
        return 0

    def get_output_shape(self):
        return self.output_size[0] * self.output_size[1] * self.n_out_channels

    def _init_constraints(self, prev_layer):
        return []

    def _test(self):
        raise NotImplementedError

    def _get_computation(self, channel_output, channel_indx):
        raise NotImplementedError

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

    def get_n_channels(self):
        """returns number of output channel of current layers

        Returns:
            int: number of output layers
        """        
        return self.n_out_channels

    def disable_compute_neurons(self):
        """routine to disable computation of nueron importance score at this layer
        """        
        self.compute_critical_neurons = False
