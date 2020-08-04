""
import torch.nn as nn
import torch.nn.functional as F
import math
from custom_pt_layers import Flatten


class AuxillaryClassifier:
    """Auxiliary classifier used for decoupled layer wise learning of convolutional networks adapted from https://github.com/eugenium/DGL
    """

    def __init__(
        self,
        feature_size=256,
        input_features=256,
        in_size=32,
        num_classes=10,
        n_lin=1,
        mlp_layers=1,
        batchn=True,
    ):
        """initialize the auxiliary network

        Args:
            feature_size (int, optional): size of the intermediate features. Defaults to 256.
            input_features (int, optional): size of the input feature map. Defaults to 256.
            in_size (int, optional): size of the input image. Defaults to 32.
            num_classes (int, optional): number of classes. Defaults to 10.
            n_lin (int, optional): number of linear transformations to the input using conv blocks. Defaults to 1.
            mlp_layers (int, optional): number of mlp classifier layers. Defaults to 1.
            batchn (bool, optional): flag to enable / disable batchnorm. Defaults to True.
        """
        self.linear_blocks = []
        self.conv_blocks = []

        self.n_lin = n_lin
        self.in_size = in_size

        if n_lin == 0 or mlp_layers == 0:
            raise NotImplementedError(
                "Check https://github.com/eugenium/DGL/blob/master/imagenet_dgl/models/auxillary_classifier.py"
            )

        feature_size = input_features
        self.conv_blocks += [
            nn.AdaptiveAvgPool2d(
                (int(math.ceil(self.in_size / 4)), int(math.ceil(self.in_size / 4)))
            )
        ]
        for n in range(self.n_lin):
            if n == 0:
                input_features = input_features
            else:
                input_features = feature_size
            self.conv_blocks += [
                nn.Conv2d(
                    input_features,
                    feature_size,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                )
            ]
            if batchn:
                self.conv_blocks += [nn.BatchNorm2d(feature_size, affine=False)]
            self.conv_blocks += [nn.ReLU(inplace=True)]
        self.conv_blocks += [
            nn.AdaptiveAvgPool2d((2, 2)),
            Flatten(),
        ]
        mlp_feat = feature_size * (2) * (2)
        for layer_indx in range(mlp_layers):
            if layer_indx == 0:
                in_feat = feature_size * 4
                mlp_feat = mlp_feat
            else:
                in_feat = mlp_feat
            self.linear_blocks += [nn.Linear(in_feat, mlp_feat)]
            if batchn:
                self.linear_blocks += [nn.BatchNorm1d(mlp_feat, affine=False)]
            self.linear_blocks += [nn.ReLU(True)]
        self.linear_blocks += [nn.Linear(mlp_feat, num_classes)]

    def get_conv_blocks(self):
        """returns convolutional blocks of this network

        Returns:
            list: list of convolutional operations in that network
        """
        return self.conv_blocks

    def get_linear_blocks(self):
        """returns mlp layers of this network

        Returns:
            list: list of linear operations in that network
        """
        return self.linear_blocks
