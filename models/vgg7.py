from torch import nn as nn
from custom_pt_layers import Flatten
from .conv_models import ConvModel

# adapted from the arc used in https://arxiv.org/abs/1605.04711


class VGG7(ConvModel):
    def __init__(
        self, name="VGG7", input_size=[32, 32], n_output_classes=10, n_channels=1
    ):
        self.name = name
        self.use_batchn = True
        self.n_output_classes = n_output_classes
        conv_module_list = []
        n_blocks = [1, 2, 2]
        m_channels_per_block = [128, 256, 512]
        input_channels = n_channels
        for channel_indx, n_out_channels in enumerate(m_channels_per_block):
            for _ in range(n_blocks[channel_indx]):
                conv_module_list += [
                    nn.Conv2d(input_channels, n_out_channels, kernel_size=3, padding=1),
                    nn.ReLU(True),
                ]
                input_channels = n_out_channels
            conv_module_list.append(nn.AvgPool2d(kernel_size=2, stride=2))
        conv_module_list.append(nn.AdaptiveAvgPool2d((2, 2)))
        conv_module_list.append(Flatten())
        first_linear_output = 1024
        linear_module_list = [
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, n_output_classes),
        ]
        super(VGG7, self).__init__(
            conv_module_list,
            linear_module_list,
            first_linear_output,
            input_size,
            n_channels,
        )
