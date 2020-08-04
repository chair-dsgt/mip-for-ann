from .sparsify_model import SparsifyModel
import numpy as np
from training import Mode
import signal
from contextlib import contextmanager
"""Backward sparsification of layers
"""

class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


class SparsifyBackward:
    def __init__(
        self, sparsify_object, mip_data_loader, n_output_classes=1,
    ):
        self.mip_data_loader = mip_data_loader
        self.n_output_classes = n_output_classes
        self.sparsify_object = sparsify_object
        self.model_train = self.sparsify_object.model_train

    def _sparsify(
        self,
        X,
        y,
        mode=Mode.MASK,
        test_original_model=True,
        test_masked_model=True,
        log_results=True,
        start_pruning_from=None,
    ):
        """calls MIP solver on input batch of data points
        
        Arguments:
            X {np.array} -- input numpy array of data points
            y {np.array} -- list of labels associated with input data points
        
        Keyword Arguments:
            mode {Mode} -- sparsification mode (Mask, Critical, Random) (default: {Mode.MASK})
            test_original_model {bool} -- flag when enabled original model will be evaluated (default: {True})
            test_masked_model {bool} -- flag when enabled sparsified model will be evaluated(default: {True})
            log_results {bool} -- flag when enabled model evaluation results will be added to the log file (default: {True})
            start_pruning_from {int} -- index of initial layer that will be represented in MIP and pruned from (default: {None})

        Returns:
            tuple(PrettyTable, float, SparsifyModel) -- returns table of evaluated model results , percentage of parameters pruned and the Sparsification Object
        """

        self.sparsify_object.get_sparsify_object().solved_mip = False
        parameters_removed_percentage = 0
        try:
            with time_limit(2 * 60 * 60):
                # 1 hours time limit
                parameters_removed_percentage = self.sparsify_object.sparsify_model(
                    X,
                    y,
                    mode=mode,
                    use_cached=False,
                    start_pruning_from=start_pruning_from,
                )
        except TimeoutException as e:
            return None
        return parameters_removed_percentage

    def sparsify_model(
        self, X, y, mode=Mode.MASK, use_cached=False, start_pruning_from=None,
    ):
        """sparsify model sequentially each class idndependently then we take the average
        
        Arguments:
            X {np.array} -- array of input batch of data points for the solver which are not used in the seq. run
            y {np.array} -- list of labels and can be safely None 
        
        Keyword Arguments:
            mode {Mode} -- mode of pruning Mask, Critical, Random (default: {Mode.MASK})
            use_cached {bool} -- bool flag to enable loading of previously cached models (default: {False})
        Returns:
            float -- percentage of removal of parameters
        """
        sparsify_base = self.sparsify_object.get_sparsify_object()
        if mode == Mode.MASK:
            initial_bounds = self.mip_data_loader.get_initial_bounds(X)
            sparsify_base.create_bounds(initial_bounds)
            n_backward_iters = len(sparsify_base.compressable_layers_indices)
            neuron_importance_score = {}
            for back_indx in range(-3, -1 * n_backward_iters - 1, -2):
                start_prune_from = sparsify_base.compressable_layers_indices[back_indx]
                parameters_removed_percentage = self._sparsify(
                    X,
                    y,
                    test_original_model=False,
                    test_masked_model=False,
                    log_results=False,
                    start_pruning_from=start_prune_from,
                )
                if parameters_removed_percentage is None:
                    self.model_train._logger.warning(
                        f"Pruning Timed out at layer {start_prune_from} breaking to show results"
                    )
                    break
                neuron_importance_score.update(sparsify_base.neuron_importance_score)
                sparsify_base.model_train.sparsify_masked = True
                sparsify_base.create_bounds(initial_bounds)
                for layer in sparsify_base.model_layers:
                    layer.update_pt_layer(
                        self.model_train.model_masked[layer.layer_indx]
                    )
                    if (
                        layer.layer_indx >= start_prune_from
                        and back_indx != -1 * n_backward_iters
                    ):
                        layer.disable_compute_neurons()
                    else:
                        layer.enable_compute_neurons()
            sparsify_base.neuron_importance_score = neuron_importance_score
            for layer in sparsify_base.model_layers:
                layer.enable_compute_neurons()
            parameters_removed_percentage = sparsify_base._filter_critical_neurons()
            sparsify_base.solved_mip = True
        else:
            parameters_removed_percentage = sparsify_base.sparsify_model(
                None, None, mode=mode, use_cached=False,
            )
        return parameters_removed_percentage

    def reset(self):
        self.sparsify_object.reset()
