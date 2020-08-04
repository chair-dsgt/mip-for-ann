from .adaptive_pool import AdaptivePool
import cvxpy as cp
import numpy as np
import copy
import torch


class AdaptiveAvgPool2d(AdaptivePool):
    """Supporting cvxpy version of adaptive pooling 2d
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
        """initializing adaptive average pooling
        
        Arguments:
            name {string} -- name of the current layer
            layer_indx {int} -- index of the current layer in the model that will be sparsified
            pytorch_layer {nn.module} -- pytorch layer 
            batch_size {int} -- size of the batch of data points fed to the MIP
        
        Keyword Arguments:
            input_size {tuple(int, int)} -- size of the input image to average pooling (default: {None})
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

    def _get_computation(self, channel_output, channel_indx):
        """returns pooling computation on current channel input
        
        Arguments:
            channel_output {cvxpy.variable} -- output from the previous layer
            channel_indx {int} -- index of the channel input to the adaptive pooling
        
        Returns:
            cvxpy.variable -- returns output of applying adaptive pooling on the input
        """
        pooled_out_list = []
        pooling_indices = self._get_pool_indices()

        def _compute_avg(block_indices):
            n_pooled_pixels = len(block_indices)
            pooled_value = (1.0 / n_pooled_pixels) * cp.sum(
                channel_output[channel_indx][:, block_indices], axis=1, keepdims=True
            )
            pooled_value = cp.reshape(pooled_value, (self.batch_size, 1))
            return pooled_value

        pooled_out_list = map(_compute_avg, pooling_indices)
        result = cp.hstack(pooled_out_list)
        return result

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
                elements = 0
                for indx in block_indx:
                    elements += current_channel[indx]
                elements /= len(block_indx)
                output_numpy[out_indx] = elements
            assert np.isclose(
                output_numpy,
                output_tensor[channel].detach().flatten().cpu().numpy(),
                atol=1e-6,
            ).all()