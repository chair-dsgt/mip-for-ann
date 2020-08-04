from torch import nn as nn
import numpy as np
import torch
from custom_pt_layers import Flatten, MaskedLinear, MaskedConv
import os
import pickle
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.sparse as sp
from sklearn.metrics import precision_recall_curve, average_precision_score
import torch.nn.functional as F
from prettytable import PrettyTable
from .logger import Logger

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from enum import Enum

"""
Sparsification mode which can be :
- Mask to remove non-critical neurons
- Random to randomly prune neurons
- Critical to prune critical neurons
"""


class Mode(Enum):
    MASK = 0
    Random = 1
    CRITICAL = 2


class PredictionResult:
    """Current Prediction result object holding stats and information of the current batch accuracy
    """

    def __init__(self, logits, targets, criterion=None):
        self.logits = logits
        self.targets = targets
        self.predictions = logits.argmax(dim=1, keepdim=True)
        self.criterion = criterion
        self.batch_size = len(targets)
        self.n_classes = int(self.logits.shape[-1])

    def get_n_correct(self):
        """get number of correctly predicted labels
        
        Returns:
            int -- number of predicted labels
        """
        return self.predictions.eq(self.targets.view_as(self.predictions)).sum().item()

    def get_data_loss(self):
        """returns criterion output value on current batch
        
        Returns:
            float -- loss value on current batch
        """
        if self.criterion is not None:
            return self.criterion(self.logits, self.targets).item()
        else:
            return -1

    def get_pred_probs(self):
        """returns softmax view on input logits
        
        Returns:
            torch.tensor -- returns prediction probabilities
        """
        return F.softmax(self.logits, dim=1)

    def get_accuracy(self):
        """compute current batch accuracy
        
        Returns:
            float -- accuracy of current batch [0-1]
        """
        n_correct = self.get_n_correct()
        return n_correct / self.batch_size

    def print_batch_probabilties(self):
        """prints batch information probabilities and predicted scores
        
        Returns:
            PrettyTable -- returns table to be added to the log file
        """
        batch_summary = PrettyTable(["Target", "Probs"])
        probabilities = self.get_pred_probs()
        for indx in range(self.batch_size):
            all_probs = [
                str(probabilities[indx, class_idnx].item())
                for class_idnx in range(self.n_classes)
            ]
            all_probs = ", ".join(all_probs)
            batch_summary.add_row([self.targets[indx].item(), all_probs])
        batch_summary.add_row(["Batch Overall Accuracy", self.get_accuracy()])
        return batch_summary


def bound_propagation(model, initial_bound):
    """propagate initial upper and lower bound batch of data through the model to get upper/lower for each layer

    Arguments:
        model {models.Model} -- trained model that will be represented by the solver
        initial_bound {tuple} -- upper and lower bound of each input data point to the solver
        start_pruning_from {int} -- index of start layer for input
    Returns:
        list -- list of upper and lower bounds for each layer
    """
    l, u = initial_bound
    bounds = []

    for layer in model:
        if isinstance(layer, Flatten):
            l_ = Flatten()(l)
            u_ = Flatten()(u)
        elif isinstance(layer, nn.Linear) or isinstance(layer, MaskedLinear):
            if isinstance(layer, MaskedLinear):
                layer = layer.get_layer()
            # check if the layer is having zero weights in case of infeasibility of bounds
            l_ = (
                layer.weight.clamp(min=0) @ l.t()
                + layer.weight.clamp(max=0) @ u.t()
                + layer.bias[:, None]
            ).t()
            u_ = (
                layer.weight.clamp(min=0) @ u.t()
                + layer.weight.clamp(max=0) @ l.t()
                + layer.bias[:, None]
            ).t()
        elif isinstance(layer, nn.Conv2d) or isinstance(layer, MaskedConv):
            if isinstance(layer, MaskedConv):
                layer = layer.get_layer()
            bias = 0
            if layer.bias is not None:
                bias = layer.bias[None, :, None, None]
            l_ = (
                nn.functional.conv2d(
                    l,
                    layer.weight.clamp(min=0),
                    bias=None,
                    stride=layer.stride,
                    padding=layer.padding,
                    dilation=layer.dilation,
                    groups=layer.groups,
                )
                + nn.functional.conv2d(
                    u,
                    layer.weight.clamp(max=0),
                    bias=None,
                    stride=layer.stride,
                    padding=layer.padding,
                    dilation=layer.dilation,
                    groups=layer.groups,
                )
                + bias
            )

            u_ = (
                nn.functional.conv2d(
                    u,
                    layer.weight.clamp(min=0),
                    bias=None,
                    stride=layer.stride,
                    padding=layer.padding,
                    dilation=layer.dilation,
                    groups=layer.groups,
                )
                + nn.functional.conv2d(
                    l,
                    layer.weight.clamp(max=0),
                    bias=None,
                    stride=layer.stride,
                    padding=layer.padding,
                    dilation=layer.dilation,
                    groups=layer.groups,
                )
                + bias
            )
        elif (
            isinstance(layer, nn.MaxPool2d)
            or isinstance(layer, nn.AvgPool2d)
            or isinstance(layer, nn.AdaptiveAvgPool2d)
        ):
            l_ = layer(l)
            u_ = layer(u)
        elif isinstance(layer, nn.BatchNorm1d) or isinstance(layer, nn.BatchNorm2d):
            if not (layer.affine):
                l_ = layer(l)
                u_ = layer(u)
            else:
                l_ = nn.functional.batch_norm(
                    l,
                    layer.running_mean,
                    layer.running_var,
                    layer.weight.clamp(min=0),
                    layer.bias,
                    False,
                    layer.momentum,
                    layer.eps,
                ) + nn.functional.batch_norm(
                    u,
                    layer.running_mean,
                    layer.running_var,
                    layer.weight.clamp(max=0),
                    layer.bias,
                    False,
                    layer.momentum,
                    layer.eps,
                )

                u_ = nn.functional.batch_norm(
                    u,
                    layer.running_mean,
                    layer.running_var,
                    layer.weight.clamp(min=0),
                    layer.bias,
                    False,
                    layer.momentum,
                    layer.eps,
                ) + nn.functional.batch_norm(
                    l,
                    layer.running_mean,
                    layer.running_var,
                    layer.weight.clamp(max=0),
                    layer.bias,
                    False,
                    layer.momentum,
                    layer.eps,
                )
        elif isinstance(layer, nn.ReLU):
            l_ = l.clamp(min=0)
            u_ = u.clamp(min=0)
        else:
            l_ = layer(l)
            u_ = layer(l)
        bounds.append((l_.detach().cpu().numpy(), u_.detach().cpu().numpy()))
        l, u = l_, u_
    return bounds


def predict(model, input_data):
    """forward pass and predict from model on input data with no_grad
    
    Arguments:
        model {torch model} -- torch model used for the prediction
        input_data {torch.tensor} -- tensor input data that we will use to predict
    
    Returns:
        torch.tensor -- predicted logits
    """
    with torch.no_grad():
        predictions = model(input_data)
        return predictions


def test_batch(model, data, target, criterion=None):
    """used to compute accuracy and loss of a batch of data points

    Arguments:
        model {Model} -- model to be tested
        data {tensor} -- batch of data points to be tested
        target {tensor} -- target labels of the input batch

    Keyword Arguments:
        criterion {nn} -- torch loss to compute model's loss on input batch (default: {None})

    Returns:
        tuple -- returns number of correct labels , loss computed based on given criterion and probabilities
    """
    output = predict(model, data)
    prediction_result = PredictionResult(output, target, criterion=criterion)
    return prediction_result


def test(model, data_loader, criterion):
    """Test results of a model on a given data loader

    Arguments:
        model {models.Model} -- pytorch model to be tested
        data_loader {torch.dataloader} -- data loader that will be tested
        criterion {nn} -- torch loss to evaluate the data

    Returns:
        tuple -- data loss using given criterion and data accuracy
    """
    model.eval()
    data_loss = 0
    correct = 0
    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        batch_result = test_batch(model, data, target, criterion)
        correct += batch_result.get_n_correct()
        data_loss += batch_result.get_data_loss()
    data_loss /= len(data_loader.dataset)
    data_accuracy = 100.0 * correct / len(data_loader.dataset)
    return data_loss, data_accuracy


def weights_init(m):
    """weight initialization for pytorch model

    Arguments:
        m {nn.module} -- module to be initialized
    """
    classname = m.__class__.__name__.lower()
    if classname.find("conv2d") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("maskedconv") != -1:
        nn.init.normal_(m.conv.weight.data, 0.0, 0.02)
    elif classname.find("batchnorm") != -1 and m.affine:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find("maskedlinear") != -1:
        nn.init.kaiming_normal_(m.linear.weight.data)
        nn.init.constant_(m.linear.bias.data, 0)
    elif classname.find("linear") != -1:
        nn.init.kaiming_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)


def get_storage_dir(config, model_name, exp_indx=0, prefix=""):
    """creates the storage directory path based on current experiment configuration

    Arguments:
        storage_parent_dir {string} -- path of the parent directory used to save experiments
        config {args} -- configuration holding argument's values for this experiment
        model_name {string} -- name of the model used in this experiment

    Keyword Arguments:
        exp_indx {int} -- for each setup we have multiple experiments with different init this to denote experiment number within the same config (default: {0})

    Returns:
        string -- save path
    """
    storage_parent_dir = config.storage_dir + prefix
    return os.path.join(
        storage_parent_dir,
        model_name,
        "lr_" + str(config.learning_rate),
        "epoch_" + str(config.epochs),
        "dataset_" + str(config.dataset),
        "optimizer_" + str(config.optimizer),
        "exp_" + str(exp_indx),
    )


def save_pickle(out_file_path, data_points):
    """pickle a set of data points

    Arguments:
        out_file_path {string} -- file path used to save the pickle dump
        data_points {object} -- data to be saved in pickle
    """
    with open(out_file_path, "wb") as handle:
        pickle.dump(data_points, handle, protocol=pickle.HIGHEST_PROTOCOL)


def adjust_learning_rate(optimizer, step, decay_boundaries=[], decay_values=[]):
    """automatic adjustment of learning rate 
    Arguments:
        optimizer {torch.optim} -- torch optimizer used during training
        step {int} -- current traint step number

    Keyword Arguments:
        decay_boundaries {list} -- list of train step boundary to apply equivalent learning rate in decay value (default: {[]})
        decay_values {list} -- values corresponding to boundaries specified in decay_boundaries (default: {[]})
    """
    if len(decay_values) == 0 or len(decay_boundaries) == 0:
        return

    decay_val_index = 0
    for boundary in decay_boundaries:
        if step >= boundary:
            decay_val_index += 1
        else:
            break
    for param_group in optimizer.param_groups:
        if step:
            param_group["lr"] = decay_values[decay_val_index]
        else:
            break


def square_indx_to_flat(x_indx, y_indx, width):
    """generate a flat index version of the input x,y points
    
    Arguments:
        x_indx {int} -- index of x axis
        y_indx {int} -- index of y axis
        width {int} -- number of elements per row
    
    Returns:
        int -- flat index
    """
    return x_indx * width + y_indx


def print_confusion_matrix(
    confusion_matrix, class_names, plt_save_dir, figsize=(10, 7), fontsize=14
):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    adapted from https://gist.github.com/shaypal5/94c53d765083101efc0240d776a23823
    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.

    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names,)
    # fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("[Exception] Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(
        heatmap.yaxis.get_ticklabels(), rotation=0, ha="right", fontsize=fontsize
    )
    heatmap.xaxis.set_ticklabels(
        heatmap.xaxis.get_ticklabels(), rotation=45, ha="right", fontsize=fontsize
    )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(plt_save_dir)


def get_pred_true_labels(model, data_loader):
    """predicts logits from input model and its equivalent labels from the data loader
    
    Arguments:
        model {torch model} -- trained model used for testing
        data_loader {torch.dataloader} -- data loader used to create its predictions from the model
    
    Returns:
        tuple(np.array, np.array) -- returns predicted logits and true labels
    """
    model.eval()
    y_test = np.array([])
    y_pred = np.array([])
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            prediction_result = test_batch(model, data, target)
            pred = prediction_result.predictions
            if y_test.shape[0] == 0:
                y_test = np.array(target.cpu().numpy())
                y_pred = np.array(pred.cpu().numpy())
            else:
                y_test = np.append(y_test, target.cpu().numpy(), axis=0)
                y_pred = np.append(y_pred, pred.cpu().numpy(), axis=0)
    return y_pred, y_test


def get_dataset_labels_to_idx(data_loader):
    """get the data index from input data loader (default data loader or custom balanced loader)
    
    Arguments:
        data_loader {torch.dataloader} -- data loader that we will use to return class vs index
    
    Returns:
        dict -- dictionary of classes to its equivalent index for input dataset
    """
    data_loader_dataset = data_loader.dataset
    if hasattr(data_loader_dataset, "dataset"):
        data_loader_dataset = data_loader_dataset.dataset
    labels_to_idx = data_loader_dataset.class_to_idx
    return labels_to_idx


def get_confusion_matrix(model, data_loader, plt_save_dir):
    """generate a confusion matrix for input model and data loader
    
    Arguments:
        model {torch model} -- model that will be evaluated and used to make confusion matrix
        data_loader {torch.dataloader} -- dataloader that will be evaluated
        plt_save_dir {string} -- save dir of the generated confusion matrix image
    """
    y_pred, y_test = get_pred_true_labels(model, data_loader)
    confusion_matrix_output = confusion_matrix(y_test, y_pred)
    labels_to_idx = get_dataset_labels_to_idx(data_loader)
    # labels_to_idx is a dict mapping label name to index
    labels = [labels_to_idx[label] for label in labels_to_idx]
    print_confusion_matrix(confusion_matrix_output, labels, plt_save_dir)


def get_precision_recall_curve(model, data_loader, plt_save_dir):
    """plotting precision recall curve
    
    Arguments:
        model {torch model} -- trained model used for evaluation
        data_loader {torch.dataloader} -- data loader that will be evaluated
        plt_save_dir {string} -- save directory of the generated precision/recall curve
    """
    y_preds, y_test = get_pred_true_labels(model, data_loader)
    labels_to_idx = get_dataset_labels_to_idx(data_loader)
    labels = [labels_to_idx[label] for label in labels_to_idx]
    precision = dict()
    recall = dict()
    average_precision = dict()
    y_test = one_hot(y_test.flatten(), len(labels))
    y_preds = one_hot(y_preds.flatten(), len(labels))
    plt.figure()
    for label in labels:
        precision[label], recall[label], _ = precision_recall_curve(
            y_test[:, label], y_preds[:, label]
        )
        average_precision[label] = average_precision_score(
            y_test[:, label], y_preds[:, label]
        )
        plt.plot(recall[label], precision[label], lw=2, label="class {}".format(label))
    average_precision["micro"] = average_precision_score(
        y_test, y_preds, average="micro"
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(
        "Average precision score, micro-averaged over all classes: AP={0:0.2f}".format(
            average_precision["micro"]
        )
    )
    plt.savefig(plt_save_dir)


def one_hot(y, k):
    """generate one hot encoded version of input labels
    
    Arguments:
        y {np.array} -- array of y labels
        k {int} -- number of classes
    
    Returns:
        np.array -- dense numpy one hot encoded array
    """
    m = len(y)
    return sp.coo_matrix((np.ones(m), (np.arange(m), y)), shape=(m, k)).todense()


def log_mean_std(logger, suffix, data):
    """writes logs of mean and std of input data
    
    Arguments:
        logger {Logger} -- python logger used to write to the file and console
        suffix {string} -- suffix name of the data to be added to the log
        data {list} -- list of data points from which we will print the stats
    """
    logger.info("[stats] {} mean {} +-/ {}".format(suffix, np.mean(data), np.std(data)))


def init_logger(storage_parent_dir, debug=True):
    """initializing logger
    
    Arguments:
        storage_parent_dir {string} -- location of logs
    
    Keyword Arguments:
        debug {bool} -- flag for debug mode (default: {True})
    
    Returns:
        Logger -- logger object used to write to the log file
    """
    return Logger.__call__(storage_parent_dir, debug).get_logger()


def log_config(logger, config):
    """Dumping experiment arguments to the log file
    
    Arguments:
        logger {Logger} -- logger used to write to the log file
        config {namespace} -- configuration of current experiment
    """
    for attr, val in config.__dict__.items():
        logger.info("[Params] {} = {}".format(str(attr), str(val)))
