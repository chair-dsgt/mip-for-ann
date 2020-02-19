from .relu import ReLUActivation
from torch import nn as nn

"""
Map used to map from torch layer to our activation layer containing its representation in cvxpy solver
"""
activations_layer_map = {
    nn.ReLU: ReLUActivation
}
