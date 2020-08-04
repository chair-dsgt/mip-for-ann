import torch.nn as nn
import torch
import copy
from .masked import Masked


class MaskedLinear(Masked):
    def __init__(self, in_dim, out_dim, indices_mask=None):
        """initialization of masked linear layer
        
        Arguments:
            nn {[type]} -- [description]
            in_dim {int} -- size of input features to linear layer
            out_dim {int} -- size of output features 
        
        Keyword Arguments:
            indices_mask {list} --  list of two lists containing indices for dimensions 0 and 1, used to create the mask, dimension 0 in_dim and dimension 1 out dim which is n neurons (default: {None})
        """
        super().__init__("linear", indices_mask)
        self.linear = nn.Linear(in_dim, out_dim)
        self.in_features = self.linear.in_features
        self.out_features = self.linear.out_features
        self.output_size = self.out_features
        self.input_size = self.in_features

    def get_layer(self):
        """used to return underlying masked layer
        
        Returns:
            nn.module -- pytorch layer being masked 
        """
        return self.linear

    @staticmethod
    def copy_layer(linear_layer, input_size=-1):
        """clone a pytorch linear layer
        
        Arguments:
            linear_layer {nn.Linear} -- linear layer that would be cloned into MaskedLinear Object
        
        Keyword Arguments:
            input_size {int} -- size of input (default: {-1})
        
        Returns:
            MaskedConv -- returns a cloned MaskedConv object of the pytorch layer
        """
        if isinstance(linear_layer, MaskedLinear):
            linear_layer = linear_layer.linear
        new_layer = MaskedLinear(linear_layer.in_features, linear_layer.out_features)
        # copy weights to the new layer
        new_layer.linear.weight.data = copy.deepcopy(linear_layer.weight.data)
        new_layer.linear.bias.data = copy.deepcopy(linear_layer.bias.data)
        return new_layer

    def mask_neurons(self, indices_mask):
        """mask neurons on the output dimension of the Linear layer based on input indices
        
        Arguments:
            indices_mask {list} -- list of indices of parameters to be masked in Linear layer
        """
        if len(indices_mask) > 0:
            self.has_indices_mask = True
        self.mask = torch.zeros([self.out_features, self.in_features]).bool()
        if indices_mask.ndim == 1:
            self.mask[indices_mask, :] = 1
        else:
            self.mask[indices_mask] = 1  # create mask
        self.mask_cached_neurons()

    def forward(self, x):
        """
        forward pass on input
        
        Arguments:
            input {Tensor} -- input image that will pass through linear layer
        
        Returns:
            tensor -- Linear layer output
        """
        self._assert_masked()
        x = self.linear(x)
        return x

    def get_sparsified_param_size(self, masked_indices):
        """computes the size of the parameters sparsified based on masked indices
        
        Arguments:
            masked_indices {np.array} -- list of indices to be masked from convolution layer (neuron index)
        
        Returns:
            int -- number of parameters that are sparsified using the input masked indices
        """
        return len(masked_indices) * self.in_features
