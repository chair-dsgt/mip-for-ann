from torch import nn as nn
from custom_pt_layers import Flatten
from .model import Model
# LEcun http://yann.lecun.com/exdb/publis/index.html#lecun-98
class FullyConnected3Model(Model):
    def __init__(self, name='FC3', input_size=784, n_output_classes=10, n_channels=-1):
        self.name = name
        module_list = [
            Flatten(),
            nn.Linear(input_size, 300),
            nn.ReLU(True),
            nn.Linear(300, 100),
            nn.ReLU(True),
            nn.Linear(100, n_output_classes)
        ] 
        super(FullyConnected3Model, self).__init__(module_list)
