from torch import nn as nn
from custom_pt_layers import Flatten
from .conv_models import ConvModel

# Lenet 5 http://yann.lecun.com/exdb/publis/index.html#lecun-98
# Interesting paper about maxpool 2d and striding https://arxiv.org/pdf/1412.6806.pdf


class ConvBaselineModel(ConvModel):
    def __init__(
        self, name="ConvBaseline", input_size=-1, n_output_classes=10, n_channels=1
    ):
        """initialization of Lenet 5 model
        
        Keyword Arguments:
            name {str} -- model name (default: {'ConvBaseline'})
            input_size {int} -- size of the  input to the model  (default: {-1})
            n_output_classes {int} -- number of output classes (default: {10})
            n_channels {int} -- number of input channels of the image (default: {1})
        """
        self.name = name
        self.n_output_classes = n_output_classes
        self.use_batchn = False
        conv_module_list = [
            nn.Conv2d(n_channels, 6, 5),
            nn.ReLU(True),
            nn.AvgPool2d((2, 2)),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(True),
            nn.AvgPool2d((2, 2)),  # replaced max pooling to average pooling
            Flatten(),
        ]
        # output of the first linear layer after convolution and input size is calculated in parent class
        first_linear_output = 120
        linear_module_list = [
            nn.ReLU(True),
            nn.Linear(120, 84),
            nn.ReLU(True),
            nn.Linear(84, n_output_classes),
        ]
        super(ConvBaselineModel, self).__init__(
            conv_module_list,
            linear_module_list,
            first_linear_output,
            input_size,
            n_channels,
        )

