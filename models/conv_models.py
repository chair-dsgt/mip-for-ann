from torch import nn as nn
from .model import Model
import torch
import numpy as np
import cv2
import torchvision
import os
from visualization import (
    GradCam,
    show_cam_on_image,
    GuidedBackpropReLUModel,
    deprocess_image,
)


class ConvModel(Model):
    def __init__(
        self,
        conv_module_list,
        linear_module_list,
        first_linear_output,
        input_size,
        n_channels,
    ):
        """Model wrapper to easily index and replace different layers of the model
        
        Arguments:
            Model {[type]} -- [description]
            conv_module_list {list} -- list of convolutional modules 
            linear_module_list {list} -- list of linear modules
            first_linear_output {int} -- output of the first linear layer after convolution to dynamically compute number of in_features
            input_size {tuple} -- size of input image
            n_channels {int} -- number of channels
        
        keywords:
            initialize {boolean} -- flag to initialize weights
        
        """
        self.input_size = input_size
        self.n_channels = n_channels
        self.final_conv_indx = len(conv_module_list) - 2  # to skip flatten module

        module_list = conv_module_list
        if first_linear_output is not None:
            flat_conv_out_size = self._compute_linear_input(conv_module_list)
            module_list.append(nn.Linear(flat_conv_out_size, first_linear_output))
        module_list += linear_module_list
        super(ConvModel, self).__init__(module_list)

    def _compute_linear_input(self, module_list):
        """computes dynamically size of input features to the fully connected part of the model
        
        Arguments:
            module_list {list} -- list of  convolutional modules that we want to compute its output size
        
        Returns:
            int -- output size of the input module list
        """
        model = nn.ModuleList(module_list)
        x = self.get_random_input(self.input_size)
        for module_pt in model:
            x = module_pt(x)
        return x.shape[1]

    def get_heat_map(
        self, tensor_image, y, save_dir, prefix="", size_upsample=(256, 256)
    ):
        """heat map computation adapted from  https://github.com/jacobgil/pytorch-grad-cam based on
        https://arxiv.org/pdf/1610.02391v1.pdf
        
        Arguments:
            tensor_image {tensor} -- tensor of the image used to compute its heat map
            y {tensor.long} -- label of input image
            save_dir {string} -- save directory of the heat maps generated
        
        Keyword Arguments:
            prefix {str} -- contains a prefix for the output file names (default: {''})
            size_upsample {tuple} -- size of the output heat map (default: {(256, 256)})
        
        Returns:
            string -- generated heat map path
        """
        tensor_image = tensor_image.unsqueeze(0).requires_grad_(True)
        heat_map_path = ""
        try:
            grad_cam = GradCam(
                model=self.model, intermediate_layer_name=[str(self.final_conv_indx)]
            )
            mask = grad_cam(tensor_image, None)
            tmp_file_name = os.path.join(
                save_dir, "tmp_original_image_{}_{}.jpg".format(str(y), prefix)
            )
            heat_map_path = os.path.join(
                save_dir, "cam_map_{}_{}.jpg".format(str(y), prefix)
            )
            torchvision.utils.save_image(tensor_image, tmp_file_name)
            img = cv2.imread(tmp_file_name, 1)
            img = np.float32(img) / 255
            show_cam_on_image(img, mask, heat_map_path)
        except Exception as e:
            heat_map_path = str(e)

        try:
            # Plotting gradient relu part
            gb_model = GuidedBackpropReLUModel(
                model=self, last_feature_indx=self.final_conv_indx
            )
            gb = gb_model(tensor_image, index=None)
            gb = gb.transpose((1, 2, 0))
            mask = cv2.resize(mask, (gb.shape[0], gb.shape[1]))
            cam_mask = cv2.merge([mask, mask, mask])

            cam_gb = deprocess_image(cam_mask * gb)
            gb = deprocess_image(gb)
            cam_gb_path = os.path.join(
                save_dir, "cam_map_gb_{}_{}.jpg".format(str(y), prefix)
            )
            cv2.imwrite(cam_gb_path, cam_gb)
        except Exception as e:
            heat_map_path += str(e)
        try:
            os.remove(tmp_file_name)
        except:
            pass
        return heat_map_path

    def get_random_input(self, input_size):
        """takes input size and generates a random torch tensor to test the model
        
        Arguments:
            input_size {int} -- size of the input
        
        Returns:
            torch.tensor -- random tensor based on the input
        """
        return torch.rand(1, self.n_channels, *input_size)
