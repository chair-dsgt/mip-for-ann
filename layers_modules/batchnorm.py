from .submodule import SubModule


class BatchNorm(SubModule):
    """parent class of cvxpy batch norm
    """

    def __init__(
        self,
        name,
        layer_indx,
        pytorch_layer,
        batch_size,
        input_size=None,
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
            compute_critical_neurons=False,
        )
        self.affine = None
        self.weights = None
        self.bias = None
        self.epsilon = None
        self.running_mean = None
        self.running_var = None
        self._extract_pt_params()

    def _extract_pt_params(self):
        self.affine = self.pytorch_layer.affine
        if self.affine:
            self.weights = (
                self.pytorch_layer.state_dict()["weight"].detach().cpu().numpy()
            )
            self.bias = self.pytorch_layer.state_dict()["bias"].detach().cpu().numpy()
        self.running_mean = (
            self.pytorch_layer.state_dict()["running_mean"].detach().cpu().numpy()
        )
        self.running_var = (
            self.pytorch_layer.state_dict()["running_var"].detach().cpu().numpy()
        )
        self.epsilon = self.pytorch_layer.eps
        self._reshape_input()

    def get_sparsified_param_size(self, masked_indices):
        return 0

    def _reshape_input(self):
        pass
