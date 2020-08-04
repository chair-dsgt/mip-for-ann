from torch import nn as nn
from custom_pt_layers import Flatten
from .model import Model


class FullyConnectedBaselineModel(Model):
    def __init__(
        self, name="FCBaseline", input_size=784, n_output_classes=10, n_channels=-1
    ):
        self.name = name
        module_list = [
            Flatten(),
            nn.Linear(input_size, 50),
            nn.ReLU(True),
            nn.Linear(50, 20),
            nn.ReLU(True),
            nn.Linear(20, n_output_classes),
        ]

        super(FullyConnectedBaselineModel, self).__init__(module_list)
