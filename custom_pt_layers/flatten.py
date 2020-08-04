from torch import nn as nn


class Flatten(nn.Module):
    """a simple module to flatten the input
    """

    def forward(self, x):
        """forward pass on input x

        Arguments:
            x {tensor} -- input that needs to be flattened
        
        Returns:
            tensor -- flattened image
        """
        return x.view(x.shape[0], -1)

