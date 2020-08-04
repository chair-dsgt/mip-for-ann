from training import Mode
from guppy import hpy


class SparsifyBase:
    def __init__(
        self,
        model_train_obj,
        sparsification_weight=5,
        threshold=1e-3,
        relaxed_constraints=False,
        mean_threshold=False,
    ):
        self.model_train = model_train_obj
        self.threshold = threshold
        self.sparsification_weight = sparsification_weight
        self._logger = model_train_obj._logger
        # used to relax relu
        self.relaxed_constraints = relaxed_constraints

        self.mean_threshold = mean_threshold

    def sparsify_model(
        self,
        input_data_flatten,
        input_data_labels,
        mode=Mode.MASK,
        use_cached=True,
        start_pruning_from=None,
        save_neuron_importance=True,
    ):
        """computes the neuron importance using solver and sparsifies the model

        Arguments:
            input_data_flatten {np.array} -- batch of input data to the solver
            input_data_labels {list} -- labels of the input batch

        Keyword Arguments:
            mode {enum} -- masking mode used (default: {Mode.MASK})
            use_cached {bool} -- flag when enabled cached solver result from previous run will be used (default: {True})
            start_pruning_from {int} -- index of initial layer that will be represented in MIP and pruned from (default: {None})

        Returns:
            float -- percentage of parameters removed
        """
        raise NotImplementedError

    def _log_memory(self):
        h = hpy()
        self._logger.info(h.heap())

    def get_sparsify_object(self):
        return self
