import cvxpy as cp
import numpy as np
from .submodule import SubModule
from scipy.linalg import toeplitz
import torch
import copy
from training.utils import square_indx_to_flat


class Conv2d(SubModule):
    """cvxpy layer of convolutional layer
    """

    def __init__(
        self,
        name,
        layer_indx,
        pytorch_layer,
        batch_size,
        input_size,
        activation=None,
        is_last_layer=False,
        compute_critical_neurons=False,
    ):
        super().__init__(
            name,
            layer_indx,
            pytorch_layer,
            batch_size,
            input_size=input_size,
            activation=activation,
            is_last_layer=is_last_layer,
            compute_critical_neurons=compute_critical_neurons,
        )
        self._prepare_conv_params()
        # setting cvxpy variable
        self.last_layer_out = None
        # so we are convolving each in_channel with separate out_channels kernel and then summing the input channels across each output kernel
        # https://medium.com/apache-mxnet/multi-channel-convolutions-explained-with-ms-excel-9bbf8eb77108
        self.layer_input = [
            cp.Variable(
                (batch_size, pytorch_layer.input_size[0] * pytorch_layer.input_size[1]),
                f"{name}_{ch_indx}",
            )
            for ch_indx in range(self.n_in_channels)
        ]
        self.neuron_importance = None
        if self.is_last_layer:  # can be used for last conv layer too
            self.last_layer_out = cp.Variable(
                (
                    batch_size,
                    self.output_size[0] * self.output_size[1] * self.n_out_channels,
                ),
                f"{name}_last",
            )
            self.compute_critical_neurons = False
        elif self.compute_critical_neurons:
            self.neuron_importance = cp.Variable(
                (self.n_out_channels), f"{name}_neuron_importance"
            )

    def _prepare_conv_params(self):
        self.n_out_channels = (
            self.pytorch_layer.out_channels
        )  # number of output feature maps
        self.n_in_channels = self.pytorch_layer.in_channels  # number of input channels
        self.kernel_size = self.pytorch_layer.kernel_size
        # we have out_channels by in_channels by each kernel converted to flat one
        self.output_size = self.pytorch_layer.output_size
        self.groups = self.pytorch_layer.groups
        self.stride = self.pytorch_layer.stride
        if type(self.stride) is not int:
            self.stride = self.stride[0]
        weights = self.pytorch_layer.conv.weight.detach().cpu().numpy()
        # for striding
        self.input_indices = []
        filter_indices = []
        if self.stride == 1:
            self.weights = self._conv_as_multiplication(
                weights, self.input_size, self.kernel_size
            )
        else:
            # prepare input indices and filter indices flattened used for unstriding conv as in n Brosch, T., Tam, R., 2014. Efficient Training of Convolutional Deep Belief Networks in the Frequency Domain for Application to High-Resolution 2D and 3D Images.
            n_unstrided_conv = self.stride ** 2
            self.input_indices = [[] for _ in range(n_unstrided_conv)]
            filter_indices = [[[], []] for _ in range(n_unstrided_conv)]
            # input indices
            max_x_indx = -1
            max_y_indx = -1
            for x_indx in range(0, self.input_size[0], self.stride):
                if x_indx + self.kernel_size[0] <= self.input_size[0]:
                    max_x_indx = x_indx + self.kernel_size[0]

            for y_indx in range(0, self.input_size[1], self.stride):
                if y_indx + self.kernel_size[1] <= self.input_size[0]:
                    max_y_indx = y_indx + self.kernel_size[0]

            for x_indx in range(self.input_size[0]):
                if x_indx >= max_x_indx:
                    continue
                for y_indx in range(self.input_size[1]):
                    if y_indx >= max_y_indx:
                        continue
                    strided_indx = self.stride * (y_indx % self.stride) + (
                        x_indx % self.stride
                    )
                    self.input_indices[strided_indx].append(
                        square_indx_to_flat(x_indx, y_indx, self.input_size[0])
                    )

            for x_indx in range(self.kernel_size[0]):
                for y_indx in range(self.kernel_size[1]):
                    strided_indx = self.stride * (y_indx % self.stride) + (
                        x_indx % self.stride
                    )
                    filter_indices[strided_indx][0].append(x_indx)
                    filter_indices[strided_indx][1].append(y_indx)

            # now computing new unstrided weights
            self.weights = []
            for stride_indx in range(n_unstrided_conv):
                # assuming input image having same w and height
                input_indices_x = [
                    indice // self.input_size[0]
                    for indice in self.input_indices[stride_indx]
                ]
                input_indices_y = [
                    indice % self.input_size[0]
                    for indice in self.input_indices[stride_indx]
                ]
                input_size = [len(set(input_indices_x)), len(set(input_indices_y))]
                current_filter_indices = tuple(filter_indices[stride_indx])
                filter_dims = [
                    len(set(current_filter_indices[0])),
                    len(set(current_filter_indices[1])),
                ]
                new_weights = np.copy(
                    weights[
                        :, :, current_filter_indices[0], current_filter_indices[1]
                    ].reshape(
                        weights.shape[0],
                        weights.shape[1],
                        filter_dims[0],
                        filter_dims[1],
                    )
                )
                self.weights.append(
                    self._conv_as_multiplication(new_weights, input_size, filter_dims,)
                )
            del weights, filter_indices, filter_dims

        if self.pytorch_layer.conv.bias is not None:
            self.bias = self.pytorch_layer.conv.bias.detach().cpu().numpy()
        else:
            self.bias = None
        if self.testing_representation:
            self._test()

    def get_constraints(self, prev_layer):
        """
        get_constraints

        Parameters:

        prev_layer_var: cvxpy variable of the layer before this one
        """
        constraints = []
        if self.activation is None:
            current_constraints = []
            for input_channel_indx in range(self.n_in_channels):
                prev_layer_computation = prev_layer.get_computation_layer(
                    input_channel_indx
                )
                upper_bound, _ = prev_layer.get_bounds(input_channel_indx)
                critical_prob = prev_layer.get_critical_neurons(input_channel_indx)
                if critical_prob is None:
                    keep_upper_bound = 0
                else:
                    keep_upper_bound = cp.multiply(1 - critical_prob, upper_bound)
                current_constraints += [
                    self.layer_input[input_channel_indx]
                    == prev_layer_computation - keep_upper_bound
                ]
            constraints += self.create_constraint(
                f"{self.name}_eq", current_constraints,
            )
        else:
            constraints += self.activation.get_constraints(self, prev_layer)
        if prev_layer.compute_critical_neurons:
            constraints += self.create_constraint(
                f"neuron_importance_bounds_{prev_layer.name}",
                [prev_layer.neuron_importance >= 0, prev_layer.neuron_importance <= 1],
            )

        if self.is_last_layer:
            if self.n_out_channels > 1:
                constraints += self.create_constraint(
                    f"{self.name}_last_layer_eq",
                    [self.last_layer_out == self._get_multi_channel_output_flat()],
                )
            else:
                constraints += self.create_constraint(
                    f"{self.name}_last_layer_multi",
                    [self.last_layer_out == self.get_computation_layer()],
                )
        return constraints

    def get_cvxpy_variable(self, channel_indx=None):
        """get the cvxpy variable associated with this layer

        Returns:
            cvxpy.variable -- cvxpy variable holding output of current layer
        """
        if channel_indx is None:
            output_channels = cp.hstack(
                [
                    self.layer_input[cur_channel_indx]
                    for cur_channel_indx in range(self.n_in_channels)
                ]
            )
        else:
            output_channels = self.layer_input[channel_indx]
        return output_channels

    def get_output_shape(self):
        """returns shape of the output from the current conv layer
        
        Returns:
            int -- flat size of teh output image from this convolution
        """
        return self.output_size[0] * self.output_size[1] * self.n_out_channels

    def get_n_neurons(self):
        """returns nymber of neurons associated with this layer
        
        Returns:
            int -- number of neurons n_channels * kh * kw
        """
        return (
            self.n_out_channels * self.kernel_size[0] * self.kernel_size[1]
        )  # used to calculate number of neurons that can be sparsified in this case it is filters

    def get_sparsified_param_size(self, masked_indices):
        """returns number of params sparsified given the input list of masked indices
        
        Arguments:
            masked_indices {np.array} -- list of indices that will be masked from current layer
        
        Returns:
            int -- number of sparsified parameters
        """
        return self.pytorch_layer.get_sparsified_param_size(masked_indices)

    def get_critical_neurons(self, channel_indx=0):
        """returns cvxpy variable of neuron importance score at specified channel index
        
        Keyword Arguments:
            channel_indx {int} -- if none returns the flat version of all channels otherwise returns neuron improtance score associated with that channel (default: {0})
        
        Returns:
            cvxpy.variable  -- cvxpy variable holding neuron importance score
        """
        if not (self.compute_critical_neurons):
            return None
        if channel_indx is None:
            output_channels = cp.hstack(
                [
                    self.get_critical_neurons(channel_indx)
                    for channel_indx in range(self.n_out_channels)
                ]
            )
            return output_channels
        else:
            critical_prob = cp.vstack(
                [
                    cp.reshape(self.neuron_importance[channel_indx], (1, 1))
                    for _ in range(self.batch_size)
                ]
            )
            critical_prob = cp.hstack(
                [
                    cp.reshape(critical_prob, (critical_prob.shape[0], 1))
                    for _ in range(self.output_size[0] * self.output_size[1])
                ]
            )
        return critical_prob

    def get_computation_layer(self, channel_indx=0):
        """returns toeplitz computation score applied at specified channel index
        
        Keyword Arguments:
            channel_indx {int} -- channel index at which the conv. operation will be applied if none returns concatenation of all channels (default: {0})
        
        Returns:
            cvxpy.variable -- output computation of current convolution layer
        """
        if channel_indx is None:
            if self.n_out_channels > 1:
                output_computation = self._get_multi_channel_output_flat()
            else:
                output_computation = self.get_computation_layer(0)
            return output_computation
        in_channels = []
        start_input_channel = 0
        end_input_channel = self.n_in_channels
        n_channels_per_group = self.n_in_channels // self.groups
        out_channels_step = self.n_out_channels // self.groups
        start_input_channel = 0
        if self.groups > 1:
            start_input_channel = (
                int(channel_indx // out_channels_step) * n_channels_per_group
            )
            end_input_channel = start_input_channel + (n_channels_per_group)
        n_weights = len(self.input_indices)  # stride >1
        for in_channel_indx in range(start_input_channel, end_input_channel):
            if n_weights > 0:
                current_channel_result = [
                    self.layer_input[in_channel_indx][
                        :, self.input_indices[weight_indx]
                    ]
                    @ self.weights[weight_indx][
                        channel_indx, in_channel_indx % n_channels_per_group
                    ].T
                    for weight_indx in range(n_weights)
                ]
                current_channel_result = cp.sum(
                    current_channel_result, keepdims=True, axis=1
                )
            else:
                current_channel_result = (
                    self.layer_input[in_channel_indx]
                    @ self.weights[
                        channel_indx, in_channel_indx % n_channels_per_group
                    ].T
                )
            in_channels.append(current_channel_result)

        if self.n_in_channels == 1:
            in_channels = in_channels[0]
        else:
            in_channels = cp.sum(in_channels, keepdims=True, axis=1)
        if self.bias is None:
            return in_channels
        return in_channels + self.bias[channel_indx]

    def get_first_layer_constraints(self, input_data):
        """returns constraints when this conv layer is the first layer (equality with input)
        
        Arguments:
            input_data {np.array} -- numpy array of input data to the solver
        
        Returns:
            list(Constraint) -- list of constraints associated with input data for first convolutional layer
        """
        constraints = []
        for param_indx, param_channel in enumerate(self.layer_input):
            constraints += self.create_constraint(
                f"{self.name}_input_eq_{param_indx}",
                [
                    param_channel
                    == input_data[:, param_indx].reshape(self.batch_size, -1)
                ],
            )
        return constraints

    def _get_multi_channel_output_flat(self):
        """used to stack output of multiple channels into one
        
        Returns:
            cvxpy.variable -- variable having stacked version of all channels output from current layer
        """
        output_channels = cp.hstack(
            [
                self.get_computation_layer(channel_indx)
                for channel_indx in range(self.n_out_channels)
            ]
        )
        return output_channels

    def _conv_as_multiplication(self, conv_weights, input_size, kernel_size):
        """converts convolutional weights into toeplitz format to make it faster and to use same constraints as the one used with the fully connected layer
        
        Arguments:
            conv_weights {np.array} -- numpy array of kernel weights in convolutional network
        """
        # inspired from https://github.com/alisaaalehi/convolution_as_multiplication/blob/master/Convolution_as_multiplication.ipynb
        output_row_num = input_size[0] + kernel_size[0] - 1
        output_col_num = input_size[1] + kernel_size[1] - 1

        output_toeplitz_row_num = self.output_size[0] * self.output_size[1]
        output_toeplitz_col_num = input_size[0] * input_size[1]
        n_out_channels = conv_weights.shape[0]
        n_in_channels = conv_weights.shape[1]
        weights = np.zeros(
            (
                n_out_channels,
                n_in_channels,
                output_toeplitz_row_num,
                output_toeplitz_col_num,
            )
        )
        for out_channel in range(n_out_channels):
            for in_channel in range(n_in_channels):
                # zero pad the filter
                # as we are computing it as cross correlation
                filter_weights = conv_weights[out_channel, in_channel, ::-1, ::-1]
                # then flipping filter upside down to avoid flipping input flat image or flipping output
                zero_padded_filter = np.pad(
                    filter_weights[::-1],
                    (
                        (output_row_num - kernel_size[0], 0),
                        (0, output_col_num - kernel_size[1]),
                    ),
                    "constant",
                    constant_values=0,
                )
                # use each row of the zero-padded F to creat a toeplitz matrix.
                #  Number of columns in this matrices are same as number of columns of input signal
                toeplitz_list = []
                # iterate from last row to the first row
                for i in range(zero_padded_filter.shape[0] - 1, -1, -1):
                    c = zero_padded_filter[i, :]  # i th row of the Filter
                    # first row for the toeplitz fuction should be defined otherwise
                    r = np.r_[c[0], np.zeros(input_size[1] - 1)]
                    # the result is wrong
                    # this function is in scipy.linalg library
                    toeplitz_m = toeplitz(c, r)
                    toeplitz_list.append(toeplitz_m)
                # doubly blocked toeplitz indices:
                #  this matrix defines which toeplitz matrix from toeplitz_list goes to which part of the doubly blocked
                c = range(1, zero_padded_filter.shape[0] + 1)
                r = np.r_[c[0], np.zeros(input_size[0] - 1, dtype=int)]
                doubly_indices = toeplitz(c, r)
                # creat doubly blocked matrix with zero values
                # shape of one toeplitz matrix
                toeplitz_shape = toeplitz_list[0].shape
                h = toeplitz_shape[0] * doubly_indices.shape[0]
                w = toeplitz_shape[1] * doubly_indices.shape[1]
                doubly_blocked_shape = [h, w]
                doubly_blocked = np.zeros(doubly_blocked_shape)

                # tile toeplitz matrices for each row in the doubly blocked matrix
                b_h, b_w = toeplitz_shape  # hight and withs of each block
                for i in range(doubly_indices.shape[0]):
                    for j in range(doubly_indices.shape[1]):
                        start_i = i * b_h
                        start_j = j * b_w
                        end_i = start_i + b_h
                        end_j = start_j + b_w
                        doubly_blocked[start_i:end_i, start_j:end_j] = toeplitz_list[
                            doubly_indices[i, j] - 1
                        ]

                row_diff = (output_row_num - self.output_size[0]) // 2
                col_diff = (output_col_num - self.output_size[1]) // 2
                if row_diff > 0 or col_diff > 0:
                    row_start_indx = row_diff
                    row_end_indx = -1 * row_diff
                    if row_diff == 0:
                        row_end_indx = output_row_num

                    col_start_indx = col_diff
                    col_end_indx = -1 * col_diff
                    if col_diff == 0:
                        col_end_indx = output_col_num
                    doubly_blocked = doubly_blocked.reshape(
                        output_row_num, output_col_num, doubly_blocked.shape[-1]
                    )[
                        row_start_indx:row_end_indx, col_start_indx:col_end_indx, :
                    ].reshape(
                        self.output_size[0] * self.output_size[1], w
                    )

                weights[out_channel, in_channel] = doubly_blocked

        return weights

    def _test(self):
        """routine used to test the current pooling implementation to make sure no discrepency between cvxpy and original pytorch layer
        """
        self.pytorch_layer.eval()
        pytorch_layer = copy.deepcopy(self.pytorch_layer).cpu()
        input_image = torch.rand(
            1, self.n_in_channels, self.input_size[0], self.input_size[1]
        )
        out_original = pytorch_layer(input_image).detach().numpy()[0]

        flat_image = input_image.cpu().numpy().squeeze().reshape(self.n_in_channels, -1)

        start_input_channel = 0
        end_input_channel = self.n_in_channels
        n_channels_per_group = self.n_in_channels // self.groups
        out_channels_step = self.n_out_channels // self.groups
        start_input_channel = 0
        n_weights = len(self.input_indices)  # stride >1
        for out_channel in range(self.n_out_channels):
            if self.groups > 1:
                if out_channel % out_channels_step == 0 and out_channel > 0:
                    start_input_channel = (
                        int(out_channel // out_channels_step) * n_channels_per_group
                    )
                end_input_channel = start_input_channel + (n_channels_per_group)
            if n_weights > 0:
                out_compute = np.sum(
                    [
                        np.sum(
                            [
                                self.weights[weight_inx][
                                    out_channel, in_chnl % n_channels_per_group
                                ]
                                @ flat_image[in_chnl][self.input_indices[weight_inx]]
                                for weight_inx in range(n_weights)
                            ],
                            axis=0,
                        )
                        for in_chnl in range(start_input_channel, end_input_channel)
                    ],
                    axis=0,
                )
            else:
                out_compute = np.sum(
                    [
                        self.weights[out_channel, in_chnl % n_channels_per_group]
                        @ flat_image[in_chnl]
                        for in_chnl in range(start_input_channel, end_input_channel)
                    ],
                    axis=0,
                )
            output_size = int(out_compute.shape[0] ** (0.5))
            out_compute = out_compute.reshape(output_size, output_size)
            bias = 0
            if self.bias is not None:
                bias = self.bias[out_channel]
            if not (
                (
                    np.isclose(
                        out_compute + bias,
                        out_original[out_channel].squeeze(),
                        atol=1e-4,
                    )
                ).all()
            ):
                error = (
                    np.square(out_compute + bias - out_original[out_channel].squeeze())
                ).mean()
                assert error < 1e-5

        # check https://discuss.pytorch.org/t/functional-conv2d-produces-different-results-vs-scipy-convolved2d/17762/2

    def get_bounds(self, channel_indx=None):
        """returns the bounds asssociated with input to this layer

        Returns:
            tuple -- upper and lower bound
        """
        if channel_indx is None:
            upper_bound = self.upper_bound.reshape(self.batch_size, -1)
            lower_bound = self.lower_bound.reshape(self.batch_size, -1)
        else:
            upper_bound = self.upper_bound[:, channel_indx, :].reshape(
                self.batch_size, -1
            )
            lower_bound = self.lower_bound[:, channel_indx, :].reshape(
                self.batch_size, -1
            )
        return upper_bound, lower_bound

    def get_n_channels(self):
        """returns number of output channels

        Returns:
            int: number of output channels
        """
        return self.n_out_channels

    def _extract_pt_params(self):
        self._prepare_conv_params()
