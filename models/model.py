from torch import nn as nn
import torch


class Model(nn.Module):
    """Model wrapper for all models that needs to be sparsified
    """

    def __init__(self, module_list):
        super(Model, self).__init__()
        self.model = nn.ModuleList(module_list)

    def __iter__(self):
        """ Returns the Iterator object """
        return iter(self.model)

    def __len__(self):
        return len(self.model)

    def __getitem__(self, index):
        return self.model[index]

    def __setitem__(self, idx, value):
        self.model[idx] = value

    def register_backward_hooks(self):
        """used to register backward hooks in case of masked modules, to updating changing masked neurons
        """
        for module_pt in self.model:
            if hasattr(module_pt, "register_masking_hooks"):
                module_pt.register_masking_hooks()

    def unregister_backward_hooks(self):
        """removing backward hook after finishing the training to avoid unnecessary extra computation
        """
        for module_pt in self.model:
            if hasattr(module_pt, "unregister_masking_hooks"):
                module_pt.unregister_masking_hooks()

    def forward(self, x):
        for module_pt in self.model:
            x = module_pt(x)
        return x

    def get_random_input(self, input_size):
        """takes input size and generates a random torch tensor to test the model
        
        Arguments:
            input_size {int} -- size of the input
        
        Returns:
            torch.tensor -- random tensor based on the input
        """
        return torch.rand(1, input_size)
