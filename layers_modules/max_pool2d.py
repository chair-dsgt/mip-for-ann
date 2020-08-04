from .pooling import Pooling
import cvxpy as cp
import numpy as np
import copy
import torch


class MaxPool2d(Pooling):
    """cvxpy of maximum pooling 2d
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
        """initializing maximum pooling
        
        Arguments:
            name {string} -- name of the current layer
            layer_indx {int} -- index of the current layer in the model that will be sparsified
            pytorch_layer {nn.module} -- pytorch layer 
            batch_size {int} -- size of the batch of data points fed to the MIP
        
        Keyword Arguments:
            input_size {tuple(int, int)} -- size of the input image to max pooling (default: {None})
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
            compute_critical_neurons=False,
        )
        self.layer_input = None
        self.z = None
        # new variable holding max elements and get computation will be returning only this variable
        self.layer_output = None
        self._update_pooling_params()

    def _init_constraints(self, prev_layer):
        """takes previous layer cvxpy object and creates cvxpy constraints for Max pool operation
        
        Arguments:
            prev_layer {SubModule} -- previous layer cvxpy
        
        Returns:
            list(constraints) -- constraints associated with max operation
        """        

        self.z = [
            cp.Variable(
                (self.batch_size, prev_layer.get_output_shape() // self.n_out_channels),
                name=self.name,
                boolean=True,
            )
            for _ in range(self.n_out_channels)
        ]

        self.layer_output = [
            cp.Variable(
                (self.batch_size, self.output_size[0] * self.output_size[1]),
                name=self.name,
            )
            for _ in range(self.n_out_channels)
        ]
        pooling_indices = self._get_pool_indices()
        constraints = []
        for channel_indx in range(self.n_out_channels):
            pooling_indx = 0
            upper_bound, lower_bound = prev_layer.get_bounds(channel_indx)
            if self.activation is not None:
                # apply activation on bounds to avoid infeasible problem
                upper_bound = self.activation.apply_numpy(upper_bound)
                lower_bound = self.activation.apply_numpy(lower_bound)
            prev_channel_output = self.layer_input[channel_indx]
            for block_indices in pooling_indices:
                x_max = self.layer_output[channel_indx][:, pooling_indx]
                z_block_elements = []
                for indx in block_indices:
                    binary_variable_z = self.z[channel_indx][:, indx]
                    z_block_elements.append(binary_variable_z)
                    umaxi = upper_bound.max(
                        axis=1,
                        where=upper_bound != upper_bound[:, indx],
                        initial=np.NINF,
                    )
                    binary_pooling_const_name = f"max_pooling_binary_{prev_layer.name}_{channel_indx}_{indx}"
                    constraints += self.create_constraint(
                        binary_pooling_const_name,
                        [
                            x_max >= prev_channel_output[:, indx],
                            x_max <= prev_channel_output[:, indx]
                            + cp.multiply(
                                (1 - binary_variable_z), umaxi - lower_bound[:, indx]
                            ),
                        ],
                    )
                constraints += self.create_constraint(
                    f"max_pooling_binary_sum_{self.name}_{channel_indx}",
                    [cp.sum(z_block_elements) == 1],
                )
                constraints += self.create_constraint(
                    f"max_pooling_bounds_{prev_layer.name}_{channel_indx}",
                    [
                        x_max >= lower_bound[:, block_indices].max(axis=1),
                        x_max <= upper_bound[:, block_indices].max(axis=1),
                    ],
                )
                pooling_indx += 1
        return constraints

    def _get_computation(self, channel_output, channel_indx):
        """returns pooling computation on current channel input
        
        Arguments:
            channel_output {cvxpy.variable} -- output from the previous layer
            channel_indx {int} -- index of the channel input to the adaptive pooling
        
        Returns:
            cvxpy.variable -- returns output of applying adaptive pooling on the input
        """
        return self.layer_output[channel_indx]

    def _test(self):
        """routine used to test the current pooling implementation to make sure no discrepency between cvxpy and original pytorch layer
        """
        self.pytorch_layer.eval()
        pytorch_layer = copy.deepcopy(self.pytorch_layer).cpu()
        input_image = torch.rand(1, self.n_out_channels, self.image_w_h, self.image_w_h)
        output_tensor = pytorch_layer(input_image)[0]
        assert self.output_size == list(output_tensor[0].shape)
        indices = self._get_pool_indices()
        for channel in range(self.n_out_channels):
            output_numpy = np.zeros(self.output_size[0] * self.output_size[1])
            current_channel = input_image[0, channel].squeeze().flatten().cpu().numpy()
            for out_indx, block_indx in enumerate(indices):
                elements = []
                for indx in block_indx:
                    elements += [current_channel[indx]]
                elements = max(elements)
                output_numpy[out_indx] = elements
            assert np.isclose(
                output_numpy, output_tensor[channel].flatten().cpu().numpy(), atol=1e-6
            ).all()