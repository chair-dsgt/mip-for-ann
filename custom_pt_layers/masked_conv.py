import torch.nn as nn
import torch
import copy
import math
from .masked import Masked
import numpy as np


class MaskedConv(Masked):
    def __init__(
        self, in_channels, out_channels, kernel_size, input_size, indices_mask=None
    ):
        """
        initialize a masked convolutional object

        Arguments:
            in_channels {int} -- number of input channels
            out_channels {int} -- number of output channels
            kernel_size {int} -- size of the kernel used by the convolutional model
            input_size {tuple} -- input image size width x height used to compute output size

        Keyword Arguments:
            indices_mask {list} --list of indices of filters that will be masked (default: {None})
        """
        super().__init__("conv", indices_mask)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = kernel_size
        self.input_size = input_size
        self.groups = self.conv.groups
        self.stride = self.conv.stride
        self._compute_output()

    def _compute_output(self):
        """
        compute the output size of this convolutional layer
        """
        self.output_size = [
            math.floor(
                (
                    (
                        self.input_size[dim_indx]
                        + 2 * self.conv.padding[dim_indx]
                        - (
                            self.conv.dilation[dim_indx]
                            * (self.kernel_size[dim_indx] - 1)
                        )
                        - 1
                    )
                    / self.conv.stride[dim_indx]
                )
                + 1
            )
            for dim_indx in range(2)
        ]
        self.out_features = (
            self.kernel_size[0] * self.kernel_size[1] * self.out_channels
        )

    def get_layer(self):
        """used to return underlying masked layer
        
        Returns:
            nn.module -- pytorch layer being masked 
        """
        return self.conv

    @staticmethod
    def copy_layer(conv_layer, input_size=(-1, -1)):
        """
        static method to copy a pytorch convolutional layer onto this masked convolutional object

        Arguments:
            conv_layer {nn.Conv2d} -- convolutional layer that will be cloned onto MaskedConv object

        Keyword Arguments:
            input_size {tuple/int} -- a tuple containing input size to this convolution layer (default: {(-1, -1)})

        Returns:
            [MaskedConv] --  returns a masked conv object with same parameters as input pytorch conv layer
        """
        if isinstance(conv_layer, MaskedConv):
            conv_layer = conv_layer.conv
        new_layer = MaskedConv(
            conv_layer.in_channels,
            conv_layer.out_channels,
            conv_layer.kernel_size,
            input_size,
        )
        # copy weights to the new layer
        new_layer.conv = copy.deepcopy(conv_layer)
        new_layer.conv.weight.data = copy.deepcopy(conv_layer.weight.data)
        if conv_layer.bias is not None:
            new_layer.conv.bias.data = copy.deepcopy(conv_layer.bias.data)
        new_layer.groups = conv_layer.groups
        new_layer.stride = conv_layer.stride
        new_layer._compute_output()
        return new_layer

    def mask_neurons(self, indices_mask):
        """
        masking entire filters of the current conv layer

        Arguments:
            indices_mask {list} -- list of filter's indices that should be masked 
        """
        if len(indices_mask) > 0:
            self.has_indices_mask = True
        self.mask = torch.zeros(
            [self.out_channels, self.in_channels // self.groups, *self.kernel_size]
        ).bool()
        if indices_mask.ndim == 1:
            self.mask[indices_mask, :, :, :] = 1
        else:
            self.mask[indices_mask] = 1  # create mask
        self.mask_cached_neurons()

    def forward(self, input):
        """
        forward pass oninput

        Arguments:
            input {Tensor} -- input image that will pass through conv layer

        Returns:
            tensor -- Convolution output
        """
        self._assert_masked()
        return self.conv(input)

    def get_sparsified_param_size(self, masked_indices):
        """computes the size of the parameters sparsified based on masked indices
        
        Arguments:
            masked_indices {np.array} -- list of indices to be masked from convolution layer (entire filter index)
        
        Returns:
            int -- number of parameters that are sparsified using the input masked indices
        """
        if masked_indices.ndim == 1:
            return (
                len(masked_indices)
                * self.kernel_size[0]
                * self.kernel_size[1]
                * self.in_channels
            )
        return np.count_nonzero(masked_indices)
