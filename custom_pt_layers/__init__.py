import torch.nn as nn
from .masked_linear import MaskedLinear
from .masked_conv import MaskedConv
from .flatten import Flatten

"""
a layer map used to map from pytorch layer to custom ;ytorch layer that supports masking neurons
"""
pytorch_layers_map = {nn.Linear: MaskedLinear, nn.Conv2d: MaskedConv}
