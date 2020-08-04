from .pooling import Pooling
import math
import torch


class AdaptivePool(Pooling):
    """Parent of adaptive pooling  layers that can be max/average of pooled pixels
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
        """initializing adaptive pooling
        
        Arguments:
            name {string} -- name of the current layer
            layer_indx {int} -- index of the current layer in the model that will be sparsified
            pytorch_layer {nn.module} -- pytorch layer 
            batch_size {int} -- size of the batch of data points fed to the MIP
        
        Keyword Arguments:
            input_size {tuple(int, int)} -- size of the input image to adaptive pooling (default: {None})
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
        self._update_pooling_params()

    def _update_pooling_params(self):
        """updates pooling params based on the pytorch layer
        """
        for i in range(2):
            # width and height
            self.output_size[i] = self.pytorch_layer.output_size[i]

    def _get_pool_indices(self):
        """returns indices of pixles that will be pooled together to redice size of the image 
        based on https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/AdaptiveAveragePooling.cpp
        
        Returns:
            list(list) -- list of all indices each item containing a list of indices that will be pooled together at that output pixel
        """
        indices = []
        sample_tensor = torch.rand(
            (1, self.n_out_channels, self.image_w_h, self.image_w_h)
        )
        w_istride = sample_tensor.stride(-1)
        h_istride = sample_tensor.stride(-2)

        for i in range(self.output_size[0]):
            h_start_index = math.floor((i * self.image_w_h) / self.output_size[0])
            h_end_index = math.ceil(((i + 1) * self.image_w_h) / self.output_size[0])
            kernel_height = h_end_index - h_start_index
            for j in range(self.output_size[1]):
                w_start_index = math.floor((j * self.image_w_h) / self.output_size[1])
                w_end_index = math.ceil(
                    ((j + 1) * self.image_w_h) / self.output_size[1]
                )
                kernel_width = w_end_index - w_start_index
                current_block_indices = []
                for ih in range(kernel_height):
                    for iw in range(kernel_width):
                        in_image_indice = (
                            h_start_index * h_istride
                            + w_start_index * w_istride
                            + ih * h_istride
                            + iw * w_istride
                        )
                        current_block_indices.append(in_image_indice)
                indices.append(current_block_indices)
        return indices