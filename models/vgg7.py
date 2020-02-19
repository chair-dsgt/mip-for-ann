from torch import nn as nn
from custom_pt_layers import Flatten
from .conv_models import ConvModel

# adapted from the arc used in https://arxiv.org/abs/1605.04711


class VGG7(ConvModel):
    def __init__(self, name='VGG7', input_size=[32, 32], n_output_classes=10, n_channels=1):
        self.name = name
        conv_module_list = [
            nn.Conv2d(n_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((2, 2)),
            Flatten()]
        first_linear_output = 1024
        linear_module_list = [
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, n_output_classes)
        ]
        super(VGG7, self).__init__(conv_module_list, linear_module_list,
                                   first_linear_output, input_size, n_channels)
