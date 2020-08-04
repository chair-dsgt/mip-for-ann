from torch import nn as nn
from custom_pt_layers import Flatten
from .model import Model

# FC-4 model


class FullyConnected4Model(Model):
    def __init__(self, name="FC4", input_size=784, n_output_classes=10, n_channels=-1):

        self.name = name
        module_list = [
            Flatten(),
            nn.Linear(input_size, 200),
            nn.ReLU(True),
            nn.Linear(200, 100),
            nn.ReLU(True),
            nn.Linear(100, 100),
            nn.ReLU(True),
            nn.Linear(100, n_output_classes),
        ]
        super(FullyConnected4Model, self).__init__(module_list)
