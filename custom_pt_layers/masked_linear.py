import torch.nn as nn
import torch
import copy


class MaskedLinear(nn.Module):
    def __init__(self, in_dim, out_dim, indices_mask=None):
        """initialization of masked linear layer
        
        Arguments:
            nn {[type]} -- [description]
            in_dim {int} -- size of input features to linear layer
            out_dim {int} -- size of output features 
        
        Keyword Arguments:
            indices_mask {list} --  list of two lists containing indices for dimensions 0 and 1, used to create the mask, dimension 0 in_dim and dimension 1 out dim which is n neurons (default: {None})
        """        
        super(MaskedLinear, self).__init__()
        if indices_mask is not None:
            self.mask_neurons(indices_mask)
        self.name = 'linear'
        self.linear = nn.Linear(in_dim, out_dim)
        self.in_features = self.linear.in_features
        self.out_features = self.linear.out_features
        self.output_size = self.out_features
        self.input_size = self.in_features
        self.handle = None
        self.has_indices_mask = False

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
        new_layer = MaskedLinear(
            linear_layer.in_features, linear_layer.out_features)
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

    def mask_cached_neurons(self):
        """
        set masked weights to zero
        """        
        if self.has_indices_mask:
            self.linear.weight.data[self.mask] = 0

    def backward_hook(self, grad):
        """
        a callback backward hook called by pytorch during gradient computation
        
        Arguments:
            grad {tensor} -- gradients of the parameter that pytorch registered this hook on
        
        Returns:
            [tensor] -- updated gradients
        """  
        # Clone due to not being allowed to modify in-place gradients
        out = grad.clone()
        out[self.mask] = 0
        return out

    def register_masking_hooks(self):
        """
        register backward hook on convolutional weights for backprop through a sparse model
        """   
        if self.has_indices_mask and self.handle is None:
            self.handle = self.linear.weight.register_hook(self.backward_hook)

    def unregister_masking_hooks(self):
        """
        unregister an existing masking hooks after finishing trainingbecause hooks would make predictions slower
        """  
        if self.handle is not None:
            self.handle.remove()

    def forward(self, x):
        """
        forward pass on input
        
        Arguments:
            input {Tensor} -- input image that will pass through linear layer
        
        Returns:
            tensor -- Linear layer output
        """  
        x = self.linear(x)
        if self.has_indices_mask:
            assert (self.linear.weight.data[self.mask] == 0).all()
        return x
