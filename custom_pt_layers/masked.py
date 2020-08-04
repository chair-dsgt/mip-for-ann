import torch.nn as nn


class Masked(nn.Module):
    def __init__(self, name, indices_mask=None):
        """parent class of masked linear and convolutions used to make a sparse version of a pytorch layer
        
        Arguments:         
            name {string} -- current layer name
        
        Keyword Arguments:
            indices_mask {numpy array} -- indices of neurons to be masked (default: {None})
        """
        super(Masked, self).__init__()
        self.name = name
        self.handle = None
        self.has_indices_mask = False
        self.mask = None
        if indices_mask is not None:
            self.has_indices_mask = True
            self.mask_neurons(indices_mask)

    def _assert_masked(self):
        """used to check if masked indices are having 0 weight value to avoid any bugs in backprop of masked
        """
        if self.has_indices_mask:
            assert (self.get_layer().weight.data[self.mask] == 0).all()

    def mask_neurons(self, indices_mask):    
        raise NotImplementedError("mask_neurons is not implemented")

    def get_layer(self):
        raise NotImplementedError("get_layer not implemented in child class")

    def mask_cached_neurons(self):   
        """
        Setting filters to be masked to zero
        """
        if self.has_indices_mask:
            self.get_layer().weight.data[self.mask] = 0
            # if self.get_layer().bias is not None:
            #     self.get_layer().bias.data[self.mask[:, 0]] = 0

    def backward_hook(self, grad):
        """
        a callback backward hook called by pytorch during gradient computation

        Arguments:
            grad {tensor} -- gradients of the parameter that pytorch registered this hook on

        Returns:
            [tensor] -- updated gradients
        """
        out = grad.clone()
        out[self.mask] = 0
        return out

    def register_masking_hooks(self):
        """
        register backward hook on convolutional weights for backprop through a sparse model
        """
        if self.has_indices_mask and self.handle is None:
            self.handle = self.get_layer().weight.register_hook(self.backward_hook)

    def unregister_masking_hooks(self):
        """
        unregister an existing masking hooks after finishing training because hooks would make predictions slower
        """
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
