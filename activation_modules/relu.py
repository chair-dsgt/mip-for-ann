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
        self.name = 'relu_' + name
        self.relaxed_constraint = relaxed_constraint
        self.v = None 

    def get_constraints(self, current_layer,  prev_layer, use_neuron_importance=True):
        """function used to get MIP constraints for the ReLU activation function
        
        Arguments:
            current_layer {layers_modules} -- takes the current layer object holding the cvxpy variables for that layer z_{i+1}
            prev_layer {[type]} -- contains previous layer object holding cvxpy variables input to current layer holding  z_{i}
        
        Keyword Arguments:
            use_neuron_importance {bool} -- a flag to enable adding neuron importance to the constraints or not, disabling this won't compute the neuron importance (default: {True})
        
        Returns:
            list -- a list of constraints that will be fed to cvxpy solver
        """
        self.v = cp.Variable(
            (current_layer.batch_size, prev_layer.get_output_shape()), name=self.name, boolean=not(self.relaxed_constraint))
        constraints = [
            current_layer.get_cvxpy_variable() >= 0
        ]
        if self.relaxed_constraint:
            constraints += [
                self.v >= 0,
                self.v <= 1
            ]
        upper_bound, lower_bound = prev_layer.get_bounds()
        prev_computation = prev_layer.get_computation_layer(None)
        if not(use_neuron_importance) or prev_layer.neuron_importance is None:
            constraints += [
                cp.multiply(
                    self.v,  upper_bound) >= current_layer.get_cvxpy_variable(),
                prev_computation >= current_layer.get_cvxpy_variable() +
                cp.multiply(1-self.v, lower_bound),
                current_layer.get_cvxpy_variable() >= prev_computation
            ]
        else:
            critical_prob = prev_layer.get_critical_neurons(None)

            if critical_prob is None:
                subtract_critical = 0
            else:
                subtract_critical = cp.multiply(1 - critical_prob, upper_bound)

            constraints += [
                cp.multiply(
                    self.v,  upper_bound) >= current_layer.get_cvxpy_variable(),
                prev_computation - subtract_critical >= current_layer.get_cvxpy_variable() +
                cp.multiply(1-self.v, lower_bound),
                current_layer.get_cvxpy_variable() >= prev_computation - subtract_critical
            ]
        return constraints
