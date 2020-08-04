
from torch import nn as nn
from custom_pt_layers import MaskedLinear, MaskedConv, Flatten
from .batchnorm_2d import BatchNorm2d
from .fully_connected import FullyConnected
from .conv_2d import Conv2d
from .avg_pool2d import AvgPool2d
from .max_pool2d import MaxPool2d
from .adaptive_avg_pool_2d import AdaptiveAvgPool2d
from .batchnorm_1d import BatchNorm1d

# layers to be ignored as they dont have anequivalent cvxpy class in layers_modules
ignored_layers_map = [Flatten, nn.ReLU, nn.Dropout]
# map of batch norm layers
batch_norm_layers = {nn.BatchNorm1d: BatchNorm1d, nn.BatchNorm2d: BatchNorm2d}

# map of pooling layers
pooling_layers = {nn.AvgPool2d: AvgPool2d, nn.MaxPool2d: MaxPool2d, nn.AdaptiveAvgPool2d: AdaptiveAvgPool2d}
# map between linear pytorch layers and cvxpy layers
linear_layers_maps = {
    MaskedLinear: FullyConnected,
    nn.Linear: FullyConnected,
}

# map between conv pytorch layers and cvxpy layers
conv_layers = {nn.Conv2d: Conv2d, MaskedConv: Conv2d}
# map between pytorch layers and cvxpy layers
layers_modules_maps = {}
layers_modules_maps.update(conv_layers)
layers_modules_maps.update(linear_layers_maps)
layers_modules_maps.update(pooling_layers)
layers_modules_maps.update(batch_norm_layers)