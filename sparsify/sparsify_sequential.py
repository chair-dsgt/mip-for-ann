from .sparsify_model import SparsifyModel
import numpy as np
from training import Mode
from .sparsify_base import SparsifyBase

"""Sequential sparsification of different classes then taking the average of the output
"""


class SparsificationStats:
    def __init__(
        self, x_list, y_list, masked_train_acc, masked_test_acc, parameters_removed_list
    ):
        self.X_list = x_list
        self.Y_list = y_list
        self.Masked_train_acc = masked_train_acc
        self.Masked_test_acc = masked_test_acc
        self.Parameters_removed_percentage_list = parameters_removed_list


class SparsifySequential(SparsifyBase):
    def __init__(
        self,
        model_train_obj,
        mip_data_loader,
        data_loaders=None,
        sparsification_weight=5,
        threshold=1e-3,
        relaxed_constraints=False,
        mean_threshold=False,
        n_output_classes=1,
    ):
        """initialization of sparsify model sequential object to run solver on each class independently the taking average

        Arguments:
            model_train_obj {ModelTrain} -- ModelTrain object used to train/test/fine tune the model
            mip_data_loader {MIPBatchLoader} -- data loader used to create batch of data fed to the MIP solver

        Keyword Arguments:
            sparsification_weight {int} -- value of the \lambda used in the loss function (default: {5})
            threshold {float} -- the value of the cutting threshold to prune neurons any neuron having a score less than this one will be pruned (default: {1e-3})
            relaxed_constraints {bool} -- a flag used to relax ReLU constraints {0,1} to continuous range [0-1] (default: {False})
            n_output_classes {int} -- integer denoting number of output classes in current model
        """
        super().__init__(
            model_train_obj,
            sparsification_weight,
            threshold,
            relaxed_constraints,
            mean_threshold,
        )
        self.mip_data_loader = mip_data_loader
        self.data_loaders = data_loaders
        self.sparsify_object = SparsifyModel(
            self.model_train,
            threshold=self.threshold,
            sparsification_weight=self.sparsification_weight,
            relaxed_constraints=self.relaxed_constraints,
        )
        self.n_output_classes = n_output_classes

    def get_sparsify_object(self):
        """returns the base sparsification object

        Returns:
            SparsifyBase: base sparsification object
        """        
        return self.sparsify_object

    def _sparsify(
        self,
        X,
        y,
        mode=Mode.MASK,
        test_original_model=True,
        test_masked_model=True,
        log_results=True,
        start_pruning_from=None,
        save_neuron_importance=True,
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
        initial_bounds = self.mip_data_loader.get_initial_bounds(X)
        self.sparsify_object = SparsifyModel(
            self.model_train,
            threshold=self.threshold,
            sparsification_weight=self.sparsification_weight,
            relaxed_constraints=self.relaxed_constraints,
        )
        self.sparsify_object.create_bounds(initial_bounds)
        parameters_removed_percentage = self.sparsify_object.sparsify_model(
            X,
            y,
            mode=mode,
            use_cached=False,
            start_pruning_from=start_pruning_from,
            save_neuron_importance=save_neuron_importance,
        )
        model_results = None
        if self.data_loaders is not None and (test_original_model or test_masked_model):
            model_results = self.model_train.print_results(
                self.data_loaders["train"],
                None,
                self.data_loaders["test"],
                test_original_model=test_original_model,
                test_masked_model=test_masked_model,
                log_results=True,
            )

        return model_results, parameters_removed_percentage

    def sparsify_sequential(
        self,
        n_images_lower=1,
        n_images_upper=10,
        random_n_images=False,
        return_stats=True,
        start_pruning_from=None,
        save_neuron_importance=True,
    ):
        """sparsify each class on its own then takes the average among multiple runs to get the neuron importance score
        
        Keyword Arguments:
            n_images_lower {int} -- lower bound of sample images used in each solver run also is the default number of samples (default: {1})
            n_images_upper {int} -- upper bound of number of images used in case of random (default: {10})
            random_n_images {bool} -- flag to randomly select number of images per class (default: {False})
            start_pruning_from {int} -- index of initial layer that will be represented in MIP and pruned from (default: {None})
        
        Returns:
            tuple(float, SparsificationStats) -- tuple of parameters removed from the current run and some stats on different solver runs
        """
        self.model_train._logger.info(
            "Running MIP solver on each class independently then taking the average"
        )
        masked_train_acc = []
        masked_test_acc = []
        parameters_removed_percentage_list = []

        X_list = []
        y_list = []
        model_layers_avg = {}
        for class_indx in range(self.n_output_classes):
            n_images = n_images_lower
            if random_n_images:
                n_images = np.random.randint(n_images_lower, n_images_upper + 1)
            self.model_train._logger.info(
                "Solving neuron importance on class {} with {} images".format(
                    class_indx, n_images
                )
            )
            X, y, _ = self.mip_data_loader.sample_from_class(class_indx, n_images)
            if return_stats:
                X_list.append(X)
                y_list.append(y)
            (masked_model_results, parameters_removed_percentage,) = self._sparsify(
                X,
                y,
                test_original_model=False,
                test_masked_model=return_stats,
                log_results=False,
                start_pruning_from=start_pruning_from,
                save_neuron_importance=False,
            )
            layer_indices = self.sparsify_object.neuron_importance_score.keys()
            for layer_indx in layer_indices:
                if layer_indx in model_layers_avg:
                    model_layers_avg[layer_indx] += (
                        self.sparsify_object.neuron_importance_score[layer_indx]
                        / self.n_output_classes
                    )
                else:
                    model_layers_avg[layer_indx] = (
                        self.sparsify_object.neuron_importance_score[layer_indx]
                        / self.n_output_classes
                    )
            if masked_model_results is not None:
                masked_train_acc.append(masked_model_results[0]["acc_train"])
                masked_test_acc.append(masked_model_results[0]["acc_test"])
            parameters_removed_percentage_list.append(parameters_removed_percentage)
        self.model_train._logger.info(
            "Creating Masked Model from Average of previous runs "
        )
        self.sparsify_object.neuron_importance_score = model_layers_avg
        parameters_removed_percentage = self.sparsify_object._filter_critical_neurons()
        self.sparsify_object.solved_mip = True
        if save_neuron_importance:
            self.sparsify_object._save_cp_layers()
        sparsification_stats = None
        if return_stats:
            sparsification_stats = SparsificationStats(
                X_list,
                y_list,
                masked_train_acc,
                masked_test_acc,
                parameters_removed_percentage_list,
            )
        return parameters_removed_percentage, sparsification_stats

    def sparsify_model(
        self,
        X,
        y,
        mode=Mode.MASK,
        use_cached=False,
        start_pruning_from=None,
        save_neuron_importance=True,
    ):
        """sparsify model sequentially each class idndependently then we take the average
        
        Arguments:
            X {np.array} -- array of input batch of data points for the solver which are not used in the seq. run
            y {np.array} -- list of labels and can be safely None 
        
        Keyword Arguments:
            mode {Mode} -- mode of pruning Mask, Critical, Random (default: {Mode.MASK})
            use_cached {bool} -- bool flag to enable loading of previously cached models (default: {False})
            start_pruning_from {int} -- index of initial layer that will be represented in MIP and pruned from (default: {None})
        Returns:
            float -- percentage of removal of parameters
        """
        if mode == Mode.MASK:
            parameters_removed_percentage, _ = self.sparsify_sequential(
                return_stats=False,
                start_pruning_from=start_pruning_from,
                save_neuron_importance=save_neuron_importance,
            )
        else:
            if self.sparsify_object is None:
                raise ValueError(
                    "You should call sparsify first to create sparsify object with mode= Mask and neuron importance pre computed"
                )
            parameters_removed_percentage = self.sparsify_object.sparsify_model(
                None,
                None,
                mode=mode,
                use_cached=False,
                start_pruning_from=start_pruning_from,
                save_neuron_importance=save_neuron_importance,
            )
        return parameters_removed_percentage

    def reset(self):
        self.sparsify_object.reset()
