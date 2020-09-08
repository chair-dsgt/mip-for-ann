
from training.utils import test_batch, log_mean_std, log_config
from sparsify import SparsifySequential
import numpy as np
from run_sparsify import prepare_arguments, prepare_images_mip_input
from train_model import (
    model_indx_is_conv,
    prepare_model_train,
    prepare_dataset,
)
from dataset import MIPBatchLoader
from enum import Enum
import torch

"""script to assesss data imbalance problem
Observations
taking average of multiple runs on different data points is almost same as running all data points in parallel
over-confident class output or data tends to prune less neurons and gives a lossless compression
confusing class output increases the search space for the mip to select which neurons are not important 
"""


def eval_confidence_function(model, X, y, confuse=False):
    """Evaluator function to compute batch score used to select input to the MIP

    Args:
        model (nn.module): input model
        X (tensor): input batch to be evaluated
        y (tensor): list of labels of input batch
        confuse (bool, optional): flag to return confusion score . Defaults to False.

    Returns:
        tuple: batch score and rprediction results object
    """
    predictions_results = test_batch(model, X, y)
    probabilities = predictions_results.get_pred_probs()
    confidence_score = 0
    epsilon = 1e-1
    for indx in range(len(y)):
        # multiplying by -1 will give you best results which is weird hahaha
        current_confidence = probabilities[indx, y[indx]].item() / (
            1 - probabilities[indx, y[indx]].item() + epsilon
        )
        if confuse:
            current_confidence = 10 - current_confidence
        confidence_score += current_confidence

    confidence_score /= len(y)
    return confidence_score, predictions_results


def eval_confusing_function(model, X, y):
    """Evaluator function to get most confusing batches as input to the MIP

    Args:
        model (nn.module): input model
        X (tensor): input batch to be evaluated
        y (tensor): list of labels of input batch.
    """
    confusion_score, predictions_results = eval_confidence_function(
        model, X, y, confuse=True
    )
    return confusion_score, predictions_results


class ExpMode(Enum):
    # average of sequential runs on different classes
    SEQUENTIAL = 0
    PARALLEL = 1


class BatchDataExperiments:
    def __init__(self, config, exp_mode, balanced=True):
        """initialize the batch data experimen tation

        Args:
            config (dict): input params disctionary
            exp_mode (ExpMode): which can be sequential or parallel (asynchronous all classes at once)
            balanced (bool, optional): flag to sample balanced batches or random number of samples per class. Defaults to True.
        """        
        self.exp_mode = exp_mode
        self.balanced = balanced
        self.config = config
        self.model_train = None
        self.mip_data_loader = None
        self.n_channels = None
        self.n_output_classes = None
        self.input_size = None
        self.n_batches = None
        self.batch_size = None
        self.data_loaders = {}
        self.model_results = []
        self.parameters_removed_percentage_list = []
        self.sparsify = None
        self._init_models()

    def _init_models(self):
        data_loaders = prepare_dataset(self.config)
        train_loader = data_loaders['train']
        val_loader = data_loaders['val']
        test_loader = data_loaders['test']
        self.data_loaders = {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader,
        }
        self.n_batches = len(val_loader)
        self.batch_size = self.config.num_samples
        self.mip_data_loader = MIPBatchLoader(
            self.config,
            val_loader,
            epsilon=1e-5,
            is_conv_model=model_indx_is_conv[config.model],
            random_order=True,
        )
        (
            X,
            y,
            initial_bounds,
            self.input_size,
            self.n_output_classes,
            self.n_channels,
        ) = prepare_images_mip_input(self.mip_data_loader)
        self.model_train = prepare_model_train(
            self.config,
            self.input_size,
            n_channels=self.n_channels,
            n_output_classes=self.n_output_classes,
            exp_indx=0,
            prefix="_batch_exp",
        )
        log_config(self.model_train._logger, self.config)
        self.model_train.train(
            train_loader, val_loader=None, num_epochs=self.config.epochs
        )
        self.model_train.print_results(
            train_loader,
            None,
            test_loader,
            test_original_model=True,
            test_masked_model=False,
        )
        self.sparsify_sequential = SparsifySequential(
            self.model_train,
            self.mip_data_loader,
            data_loaders=self.data_loaders,
            sparsification_weight=self.config.sparsification_weight,
            threshold=self.config.threshold,
            relaxed_constraints=self.config.relaxed,
            n_output_classes=self.n_output_classes,
        )

    def log_info(self, message):
        """add a message to the log file

        Args:
            message (str): message that needs to be logged
        """        
        self.model_train._logger.info(message)

    def set_priority_function_mip(self, eval_function):
        """set evaluator used to select the input batch to the MIP

        Args:
            eval_function (function): function that takes batch as input and returns a priority score
        """        
        self.mip_data_loader.set_evaluator_function(eval_function)
        self.mip_data_loader.set_model(self.model_train.model)

    def eval_batch(self, X, y):
        """computes priority score to the input batch

        Args:
            X (tensor): input data points to the MIP
            y (tensor): list of labels associated with this batch
        """        
        if self.mip_data_loader.eval_function is not None:
            batch_score, predictions_results = self.mip_data_loader.eval_function(
                self.model_train.model, X, y
            )
            self.log_info("Current Batch Evaluation score ==> {}".format(batch_score))

    def run_sequential(self):
        """Runs the sequential experiments, computes importance score for each class and then computes the average for all classes

        Returns:
            tuple: list of model results of sequential run and percentage of paramaters removed
        """        
        (
            parameters_removed_percentage,
            sequential_stats,
        ) = self.sparsify_sequential.sparsify_sequential(
            n_images_upper=self.config.num_samples, random_n_images=not (self.balanced)
        )

        self.model_train._logger.info("Stats of different data points")
        log_mean_std(
            self.model_train._logger,
            "Masked Train Acc",
            sequential_stats.Masked_train_acc,
        )
        log_mean_std(
            self.model_train._logger,
            "Masked Test Acc",
            sequential_stats.Masked_test_acc,
        )
        log_mean_std(
            self.model_train._logger,
            "Pruning Percentage",
            sequential_stats.Parameters_removed_percentage_list,
        )
        self.model_train._logger.info(
            "[Exp] Model Results of average of multiple independent runs on different classes"
        )
        model_results = self.model_train.print_results(
            self.data_loaders["train"],
            None,
            self.data_loaders["test"],
            test_original_model=True,
            test_masked_model=True,
        )
        if self.balanced:
            self.model_train._logger.info(
                "[Exp] Parallel Run on same sampled data points"
            )
            X = torch.cat(sequential_stats.X_list, 0)
            y = np.concatenate(sequential_stats.Y_list, axis=0)
            (
                parallel_model_results,
                parameters_removed_parallel,
            ) = self.sparsify_sequential._sparsify(
                X, y, test_original_model=False, test_masked_model=True
            )
            model_results.append(parallel_model_results)
            parameters_removed_percentage = [
                parameters_removed_percentage,
                parameters_removed_parallel,
            ]
        return model_results, parameters_removed_percentage

    def run_parallel(self):
        """Runs asynchronous all classes at once

        Returns:
            tuple: model results and percentage of sparsification
        """        
        X, y, _ = next(self.mip_data_loader)
        self.eval_batch(X, y)
        (
            model_results,
            parameters_removed_percentage,
        ) = self.sparsify_sequential._sparsify(
            X, y, test_original_model=True, test_masked_model=True
        )
        return model_results, parameters_removed_percentage

    def run_experiment(self):
        """Starts the experiment

        Returns:
            tuple: model results and percentage of sparsification
        """        
        self.model_train._logger.info(
            "[Exp] Started Model {} batch data experiments by changing input to MIP with mode {} and balanced ={}".format(
                self.model_train.model.name, self.exp_mode.name, self.balanced
            )
        )
        if self.exp_mode == ExpMode.SEQUENTIAL:
            return self.run_sequential()
        return self.run_parallel()

    def append_results(self, model_results, parameters_removed_percentage):
        self.model_results.append(model_results)
        self.parameters_removed_percentage_list.append(parameters_removed_percentage)

    def clear_stats(self):
        self.model_results = []
        self.parameters_removed_percentage_list = []

    def print_stats(self):
        """print statistics of the current experiment
        """        
        self.model_train._logger.info("Experiments Statistics ")
        original_model_train_acc = []
        original_model_test_acc = []
        masked_train_acc = []
        masked_test_acc = []
        parallel_vs_sequential_enabled = self.exp_mode == ExpMode.SEQUENTIAL and self.balanced
        parallel_train_acc = []
        parallel_test_acc = []
        paralel_parameters_removed = []
        sequential_paramemters_removed = []
        for result_indx, model_result in enumerate(self.model_results):
            original_model_train_acc.append(model_result[0]["acc_train"])
            original_model_test_acc.append(model_result[0]["acc_test"])
            masked_train_acc.append(model_result[1]["acc_train"])
            masked_test_acc.append(model_result[1]["acc_test"])
            if parallel_vs_sequential_enabled:
                parallel_train_acc.append(model_result[2][0]["acc_train"])
                parallel_test_acc.append(model_result[2][0]["acc_test"])
                parameters_removed_percentage = self.parameters_removed_percentage_list[
                    result_indx
                ]
                sequential_paramemters_removed.append(parameters_removed_percentage[0])
                paralel_parameters_removed.append(parameters_removed_percentage[1])
        self._print_mean_std(
            original_model_train_acc,
            original_model_test_acc,
            masked_train_acc,
            masked_test_acc,
            sequential_paramemters_removed,
            parallel_train_acc,
            parallel_test_acc,
            paralel_parameters_removed,
        )

    def _print_mean_std(
        self,
        original_model_train_acc,
        original_model_test_acc,
        masked_train_acc,
        masked_test_acc,
        sequential_paramemters_removed,
        parallel_train_acc,
        parallel_test_acc,
        paralel_parameters_removed,
    ):
        log_mean_std(
            self.model_train._logger, "Original Train Acc", original_model_train_acc,
        )
        log_mean_std(
            self.model_train._logger, "Original Test Acc", original_model_test_acc
        )
        log_mean_std(
            self.model_train._logger, "Masked Train Acc", masked_train_acc,
        )
        log_mean_std(self.model_train._logger, "Masked Test Acc", masked_test_acc)
        parallel_vs_sequential_enabled = len(sequential_paramemters_removed) > 0
        if parallel_vs_sequential_enabled:
            log_mean_std(
                self.model_train._logger,
                "Sequential Pruning Percentage",
                sequential_paramemters_removed,
            )
            log_mean_std(
                self.model_train._logger,
                "Parallel Masked Train Acc",
                parallel_train_acc,
            )
            log_mean_std(
                self.model_train._logger, "Parallel Masked Test Acc", parallel_test_acc
            )
            log_mean_std(
                self.model_train._logger,
                "Parallel Pruning Percentage",
                paralel_parameters_removed,
            )
        else:
            log_mean_std(
                self.model_train._logger,
                "Pruning Percentage",
                self.parameters_removed_percentage_list,
            )


def prepare_config():
    parser = prepare_arguments()
    parser.add_argument("--n-experiments", "-nex", default=1, type=int)
    parser.add_argument("--balanced-batch", "-bbm", action="store_true")
    parser.add_argument("--parallel-exp", "-ppexp", action="store_true")
    config = parser.parse_args()
    return config


if __name__ == "__main__":
    config = prepare_config()
    # Set experiment mode
    if config.parallel_exp:
        exp_mode = ExpMode.PARALLEL
    else:
        exp_mode = ExpMode.SEQUENTIAL
    n_experiments = config.n_experiments
    balanced = config.balanced_batch
    exp = BatchDataExperiments(config, exp_mode, balanced=balanced)
    exp.log_info(
        "[Exp] Set Confidence eval function to prioritize high confident mip batches"
    )
    exp.set_priority_function_mip(eval_confidence_function)
    for i in range(n_experiments):
        exp.log_info("[Exp] Started Experiment {}".format(i))
        model_results, parameters_removed_percentage = exp.run_experiment()
        exp.append_results(model_results, parameters_removed_percentage)
    exp.print_stats()
    if exp_mode == ExpMode.PARALLEL:
        exp.clear_stats()
        exp.log_info(
            "[Exp] Set Confusion eval function to prioritize confusing mip batches"
        )
        exp.set_priority_function_mip(eval_confusing_function)
        for i in range(n_experiments):
            model_results, parameters_removed_percentage = exp.run_experiment()
            exp.append_results(model_results, parameters_removed_percentage)

        exp.print_stats()
