import numpy as np
import cvxpy as cp
from .activation import Activation


class ReLUActivation(Activation):
    def __init__(self, name, relaxed_constraint=False):
        """Initialization of ReLU activation object holding cvxpy variables

        Arguments:
            name {string} -- holding current relu name as an identifier

        Keyword Arguments:
            relaxed_constraint {bool} -- a flag to relax the ReLU constraints v_i instead of {0,1} to continuous range [0,1] (default: {False})
        """
        super().__init__(name, relaxed_constraint)
        self.name = "relu_" + name
        self.relaxed_constraint = relaxed_constraint
        self.v = None  # variable used as discussed in https://adversarial-ml-tutorial.org/adversarial_examples/

    def get_constraints(self, current_layer, prev_layer, use_neuron_importance=True):
        """function used to get MIP constraints for the ReLU activation function

        Arguments:
            current_layer {layers_modules} -- takes the current layer object holding the cvxpy variables for that layer z_{i+1}
            prev_layer {[type]} -- contains previous layer object holding cvxpy variables input to current layer holding  z_{i}

        Keyword Arguments:
            use_neuron_importance {bool} -- a flag to enable adding neuron importance to the constraints or not, disabling this won't compute the neuron importance (default: {True})

        Returns:
            list -- a list of constraints that will be fed to cvxpy solver
        """
        prev_computation = prev_layer.get_computation_layer(None)
        current_layer_var = current_layer.get_cvxpy_variable()
        compute_neuron_importance = (
            use_neuron_importance and prev_layer.compute_critical_neurons
        )
        if (
            self.v is not None
            and self.v.value is not None
            and not (compute_neuron_importance)
        ):
            relu_mask = self.v.value
            constraints = current_layer.create_constraint(
                self.name + "_hasv_value",
                [current_layer_var == cp.multiply(prev_computation, relu_mask)],
            )
            return constraints
        else:
            self.v = cp.Variable(
                (current_layer.batch_size, prev_layer.get_output_shape()),
                name=self.name,
                boolean=not (self.relaxed_constraint),
            )
        constraints = current_layer.create_constraint(
            "greater_than_zero_" + self.name, [current_layer_var >= 0]
        )
        if self.relaxed_constraint:
            constraints += current_layer.create_constraint(
                self.name + "_relaxed_relu", [self.v >= 0, self.v <= 1]
            )

        upper_bound, lower_bound = prev_layer.get_bounds()
        if not (compute_neuron_importance):
            constraints += current_layer.create_constraint(
                self.name + "_no_neuronimportance",
                [
                    current_layer_var <= cp.multiply(self.v, self.apply_numpy(upper_bound)),
                    current_layer_var >= prev_computation,
                    current_layer_var
                    <= prev_computation - cp.multiply(1 - self.v, lower_bound),
                ],
            )
        else:
            critical_prob = prev_layer.get_critical_neurons(None)

            if critical_prob is None:
                subtract_critical = 0
            else:
                subtract_critical = cp.multiply(
                    1 - critical_prob, self.apply_numpy(upper_bound)
                )

            constraints += current_layer.create_constraint(
                self.name + "_neurons_inequality",
                [
                    current_layer_var <= cp.multiply(self.v, self.apply_numpy(upper_bound)),
                    current_layer_var >= prev_computation - subtract_critical,
                    current_layer_var
                    <= prev_computation
                    - cp.multiply(1 - self.v, lower_bound)
                    - subtract_critical,
                ],
            )
        return constraints

    def apply_numpy(self, input_array):
        """Applies activation relu on input numpy array

        Args:
            input_array (np.array): numpy array on which the activation will be applied

        Returns:
            np.array: output of relu operation max(0, input_array)
        """        
        return np.maximum(input_array, 0)
