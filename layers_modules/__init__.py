from .fully_connected import FullyConnected
from custom_pt_layers import MaskedLinear, Flatten
from torch import nn as nn


# layers to be ignored as they dont have anequivalent cvxpy class in layers_modules
ignored_layers_map = [Flatten, nn.ReLU, nn.Dropout, nn.MaxPool2d]

# map between linear pytorch layers and cvxpy layers
linear_layers_maps = {
    MaskedLinear: FullyConnected,
    nn.Linear: FullyConnected,
}

# map between conv pytorch layers and cvxpy layers
conv_layers = {
}
# map between pytorch layers and cvxpy layers
layers_modules_maps = {}
layers_modules_maps.update(conv_layers)
layers_modules_maps.update(linear_layers_maps)
