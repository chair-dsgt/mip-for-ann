
class Activation:
    def __init__(self, name, relaxed_constraint=False):
        """[summary]
        
        Arguments:
            name {string} -- contains the name of the activation (as an identifier)
        
        Keyword Arguments:
            relaxed_constraint {bool} -- a flag to relax the ReLU constraints v_i instead of {0,1} to continuous range [0,1] (default: {False})
        """        
        self.name = name
        self.relaxed_constraint = relaxed_constraint

    def get_constraints(self, current_layer,  prev_layer, use_neuron_importance=True):
        raise NotImplementedError
