from .constraint import Constraint
import copy


class SubModule:
    def __init__(
        self,
        name,
        layer_indx,
        pytorch_layer,
        batch_size,
        input_size=None,
        activation=None,
        is_last_layer=False,
        compute_critical_neurons=True,
    ):
        self.name = name
        self.layer_indx = layer_indx
        self.batch_size = batch_size
        self.pytorch_layer = pytorch_layer
        self.lower_bound = None
        self.upper_bound = None
        self.activation = activation
        self.input_size = input_size  # image height x width
        self.is_last_layer = is_last_layer
        self.last_layer_out = None
        self.layer_out = None
        self.compute_critical_neurons = compute_critical_neurons
        self._bak_neuron_importance = None
        self.testing_representation = False

    def set_bounds(self, lower, upper):
        """setting bounds of current layer
        
        Arguments:
            lower {np.array} -- contains the lower bound for the input to current layer
            upper {np.array} -- contains the upper bound for the input to current layer
        """
        if lower.ndim > 3:
            # for convnets
            lower = lower.reshape(lower.shape[0], lower.shape[1], -1)
        if upper.ndim > 3:
            # for convnets
            upper = upper.reshape(lower.shape[0], lower.shape[1], -1)
        self.lower_bound = lower
        self.upper_bound = upper

    def get_constraints(self, prev_layer):
        raise NotImplementedError

    def get_cvxpy_variable(self, channel_indx=None):
        raise NotImplementedError

    def get_output_shape(self):
        raise NotImplementedError

    def get_n_neurons(self):
        raise NotImplementedError

    def get_layer_out(self):
        if self.is_last_layer:
            return self.last_layer_out
        return self.layer_out

    def get_computation_layer(self, channel_indx=0):
        # used to calculate output based on weight and bias values
        raise NotImplementedError

    def get_critical_neurons(self, channel_indx=0):
        return None

    def get_first_layer_constraints(self, input_data):
        raise NotImplementedError

    def get_sparsified_param_size(self, masked_indices):
        raise NotImplementedError

    def get_bounds(self):
        raise NotImplementedError

    def get_n_channels(self):
        raise NotImplementedError

    def update_pt_layer(self, pt_layer):
        self.pytorch_layer = pt_layer
        self._extract_pt_params()

    def _extract_pt_params(self):
        pass

    def disable_compute_neurons(self):
        """a routine to disable computation of neuron importance score
        """
        self.compute_critical_neurons = False
        if hasattr(self, "neuron_importance") and self.neuron_importance is not None:
            self._bak_neuron_importance = copy.copy(self.neuron_importance)
            self.neuron_importance = None

    def enable_compute_neurons(self):
        """routine to re-enable computation of neuron importance score
        """
        if hasattr(self, "neuron_importance"):
            self.compute_critical_neurons = True
            if self._bak_neuron_importance is not None:
                self.neuron_importance = copy.copy(self._bak_neuron_importance)

    def create_constraint(self, name, constraints):
        """creates a constraint object

        Args:
            name (str): name of this set of constraints
            constraints (list): list of constraints

        Returns:
            list: list of created constraints
        """
        return [Constraint(name, constraints)]
