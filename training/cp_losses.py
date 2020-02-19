import cvxpy as cp
import numpy as np
import scipy.sparse as sp


def one_hot(y, k):
    m = len(y)
    return sp.coo_matrix((np.ones(m), (np.arange(m), y)), shape=(m, k)).todense()


def softmax_loss(last_layer_logits, y):
    """marginal softmax based on https://ttic.uchicago.edu/~kgimpel/papers/gimpel+smith.naacl10.pdf
    
    Arguments:
        last_layer_logits {cvxpy variable} -- decision variable of the solver approximated output to the model's logits
        y {np.array} -- labels of input batch to the solver
    Returns:
        cvxpy variable -- the loss computed  used as the solver's objective
    """     
    k = last_layer_logits.shape[1]
    Y = one_hot(y, k)
    return (cp.sum(cp.log_sum_exp(last_layer_logits, axis=1)) -
            cp.sum(cp.multiply(Y, last_layer_logits)))
