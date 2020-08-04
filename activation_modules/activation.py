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

    def get_constraints(self, current_layer, prev_layer, use_neuron_importance=True):
        """Takes current cvxpy layer and previous layer to return list of cvxpy constraints
        
        Arguments:
            current_layer {layers_modules.} -- layer module containing cvxpy variables
            prev_layer {layers_modules} -- previous layer module with cvxpy variables
        
        Keyword Arguments:
            use_neuron_importance {bool} -- flag to add neuron importance score to the constraint or not (default: {True})
        
        Raises:
            NotImplementedError: In case child class doesn't override this funciton it will raise not implemented error
        """
        raise NotImplementedError

    def apply_numpy(self, input_array):
        """Apply numpy operation same as the activation function
        
        Arguments:
            input_array {np.array} -- numpy array input that the activation will be applied on
        
        Raises:
            NotImplementedError: In case child class doesn't override this funciton it will raise not implemented error
        """
        raise NotImplementedError
    