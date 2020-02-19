

class SubModule:
    def __init__(self,name, layer_indx, pytorch_layer, batch_size, input_size=None,activation=None,is_last_layer=False, compute_critical_neurons=True):
        self.name = name
        self.layer_indx = layer_indx
        self.batch_size = batch_size
        self.pytorch_layer = pytorch_layer
        self.lower_bound = None
        self.upper_bound = None
        self.activation = activation  
        self.input_size = input_size # image height x width
        self.is_last_layer = is_last_layer 
        self.last_layer_out = None
        self.layer_out = None
        self.compute_critical_neurons = compute_critical_neurons
    
    def set_bounds(self, l, u):
        """setting bounds of current layer
        
        Arguments:
            l {np.array} -- contains the lower bound for the input to current layer
            u {np.array} -- contains the upper bound for the input to current layer
        """        
        if l.ndim >3:
            # for convnets
            l = l.reshape(l.shape[0],l.shape[1],-1)
        if u.ndim >3:
            # for convnets
            u = u.reshape(l.shape[0],l.shape[1],-1)
        self.lower_bound = l
        self.upper_bound = u
    
    def get_constraints(self):
        raise NotImplementedError

    def get_cvxpy_variable(self):
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
        raise NotImplementedError

    def get_first_layer_constraints(self, input_data):
        raise NotImplementedError

    def get_sparsified_param_size(self, masked_indices):
        raise NotImplementedError
 
    def get_bounds(self):
        raise NotImplementedError

    




