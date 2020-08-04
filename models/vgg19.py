from torch import nn as nn
from custom_pt_layers import Flatten
from .conv_models import ConvModel

"""https://arxiv.org/pdf/1409.1556.pdf vgg-19
"""


class VGG19(ConvModel):
    def __init__(
        self, name="VGG19", input_size=[32, 32], n_output_classes=10, n_channels=1
    ):
        self.name = name
        self.use_batchn = True
        self.n_output_classes = n_output_classes
        conv_module_list = []
        n_blocks = [2, 2, 4, 4, 4]
        m_channels_per_block = [64, 128, 256, 512, 512]
        input_channels = n_channels
        for channel_indx, n_out_channels in enumerate(m_channels_per_block):
            for _ in range(n_blocks[channel_indx]):
                conv_module_list += [
                    nn.Conv2d(input_channels, n_out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(n_out_channels),
                    nn.ReLU(True),
                ]
                input_channels = n_out_channels
            conv_module_list.append(nn.MaxPool2d(kernel_size=2, stride=2))
        conv_module_list.append(nn.AdaptiveAvgPool2d((7, 7)))
        conv_module_list.append(Flatten())

        first_linear_output = 4096
        linear_module_list = [
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, n_output_classes),
        ]
        super(VGG19, self).__init__(
            conv_module_list,
            linear_module_list,
            first_linear_output,
            input_size,
            n_channels,
        )
