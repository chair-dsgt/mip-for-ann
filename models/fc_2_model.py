from torch import nn as nn
from custom_pt_layers import Flatten
from .model import Model


class FullyConnected2Model(Model):
    def __init__(self, name="FC2", input_size=784, n_output_classes=10, n_channels=-1):
        self.name = name
        module_list = [
            Flatten(),
            nn.Linear(input_size, 1000),
            nn.ReLU(True),
            nn.Linear(1000, n_output_classes),
        ]
        super(FullyConnected2Model, self).__init__(module_list)
