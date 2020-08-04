from torch import nn as nn
from custom_pt_layers import Flatten
from .conv_models import ConvModel

"""https://arxiv.org/pdf/1409.1556.pdf vgg-16 version d with an update as in 
    Differences:
        The number of parameters in conv layers are the same as the original.
        The number of parameters in fc layers are reduced to 512 (4096 -> 512).
        The number of total parameters are different, not just because of the size of fc layers,
        but also due to the fact that the first fc layer receives 1x1 image rather than 7x7 image
        because the input is CIFAR not IMAGENET.
        No dropout is used. Instead, batch norm is used.
    Other refereneces.
        (1) The original paper:
        - paper: https://arxiv.org/pdf/1409.1556.pdf
        - code: http://www.robots.ox.ac.uk/~vgg/research/very_deep/
        * Dropout between fc layers.
        * There is no BatchNorm.
        (2) VGG-like by Zagoruyko, adapted for CIFAR-10.
        - project and code: http://torch.ch/blog/2015/07/30/cifar.html
        * Differences to the original VGG-16 (1):
            - # of fc layers 3 -> 2, so there are 15 (learnable) layers in total.
            - size of fc layers 4096 -> 512.
            - use BatchNorm and add more Dropout.
"""


class VGG16NoBn(ConvModel):
    def __init__(
        self, name="VGG16NoBn", input_size=[32, 32], n_output_classes=10, n_channels=1
    ):
        self.name = name
        self.use_batchn = False
        self.n_output_classes = n_output_classes
        conv_module_list = []
        n_blocks = [2, 2, 3, 3, 3]
        m_channels_per_block = [64, 128, 256, 512, 512]
        input_channels = n_channels
        for channel_indx, n_out_channels in enumerate(m_channels_per_block):
            for _ in range(n_blocks[channel_indx]):
                conv_module_list += [
                    nn.Conv2d(input_channels, n_out_channels, kernel_size=3, padding=1),
                    nn.ReLU(True),
                ]
                input_channels = n_out_channels
            conv_module_list.append(nn.AvgPool2d(kernel_size=2, stride=2))
        conv_module_list.append(Flatten())

        first_linear_output = 512
        linear_module_list = [
            nn.ReLU(True),
            nn.Linear(512, n_output_classes),
        ]
        super(VGG16NoBn, self).__init__(
            conv_module_list,
            linear_module_list,
            first_linear_output,
            input_size,
            n_channels,
        )


