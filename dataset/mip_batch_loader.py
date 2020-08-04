from training.utils import test_batch, device
import numpy as np
import math
import torch
import torch.nn.functional as F


class CustomLoss(torch.nn.Module):
    """Custom loss of weighted accuracy giving more weight to correctly predicted with lower probs.
    """    
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets, nb_classes=None):
        """computes the value of its loss on input logits and targets

        Args:
            logits (tensor): input batch of data points
            targets (tensor): labels for the input batch
            nb_classes (int, optional): number of classes. Defaults to None.

        Returns:
            tensor: weighted accuracy value
        """        
        pred = logits.argmax(dim=1, keepdim=True) 
        probs = F.softmax(logits, dim=1)
        n_correct = ((1 - probs) * pred.eq(targets.view_as(pred))).sum()
        return n_correct / logits.shape[0]


def evaluator(model, X, y):
    """default evaluator based on weighted accuracy to get correctly predicted with lower probs. to allow sparsification

    Args:
        model (nn.module): model represented by the MIP
        X (tensor.float): input batch to the MIP
        y (tensor.long): labels of input batch

    Returns:
        tuple: evaluation of the batch value, results object of that prediction
    """    
    # default evaluator based on weighted accuracy to get correctly predicted with lower probs. to allow sparsification
    prediction_results = test_batch(model, X, y, criterion=CustomLoss())
    return prediction_results.get_data_loss(), prediction_results


class MIPBatchLoader:
    """used to sample batches of data points as input to the MIP solver
    """

    def __init__(
        self,
        config,
        data_loader,
        is_conv_model=False,
        epsilon=1e-5,
        model=None,
        random_order=False,
    ):
        """initialize a mip data loader
        
        Arguments:
            config {namespace} -- contains arguments passed to run_sparsify
            data_loader {BalancedDataLoader} -- data loader used to sample data points from
        
        Keyword Arguments:
            is_conv_model {bool} -- a flag enabled for convolution models (default: {False})
            epsilon {float} -- epsilon between upper and lower bounds (default: {1e-5})
            model {torch model} -- model used to evaluate batches and to get high accuracy batches as input to MIp (default: {None})
            random_order {bool} -- next batch of data point is sampled randomly from data loader (default: {False})
        """
        self.config = config
        self.epsilon = epsilon
        self.model = model
        self.data_loader = data_loader
        self.eval_function = evaluator
        self.random_order = random_order
        self.batch_size = data_loader.batch_size
        self.n_batches = len(data_loader)
        self.n_mip_batches_in_loader_batch = int(
            math.floor(self.batch_size / config.num_samples)
        )
        self.is_conv_model = is_conv_model
        self._reset()

    def set_model(self, model):
        """set the evaluator model and reset the index
        
        Arguments:
            model {torch model} -- model that needs to be sparsified set as an evaluator in Mip data loader
        """
        self.model = model
        self._reset()

    def update_data_loader(self, data_loader):
        """updating used data loader
        
        Arguments:
            data_loader {BalancedDataLoader} -- a data loader used to sample MIP data points
        """
        self.data_loader = data_loader
        self._reset()

    def _reset(self):
        """reset sampling index to the start
        """
        self.start_indx = 0
        if self.random_order:
            self.batch_indx = np.random.randint(0, self.n_batches)
        self.batch_indx = 0
        self.end_indx = self.config.num_samples + self.start_indx

    def set_evaluator_function(self, eval_function):
        """setting lambda function used to order mip batches and sampling one with highest balue
        
        Arguments:
            eval_function {lambda} -- function that returns a priority score and takes the batch, the higher score the more probable it will be sampled
        """
        self.eval_function = eval_function
        self._reset()

    def _increment_indices(self):
        """updating index on each sampling iteration
        """
        if self.end_indx >= self.batch_size:
            self.start_indx = 0
            if self.random_order:
                self.batch_indx = np.random.randint(0, self.n_batches)
            else:
                self.batch_indx += 1
        else:
            self.start_indx = self.end_indx
        if self.random_order:
            self.start_indx = np.random.randint(0, self.n_mip_batches_in_loader_batch)
        self.end_indx = self.start_indx + self.config.num_samples
        if self.batch_indx >= self.n_batches:
            self._reset()

    def get_initial_bounds(self, X):
        """computing initial upper and lower bounds with epsilon difference
        
        Arguments:
            X {torch.tensor} -- tensor of input data points passed to MIP
        
        Returns:
            tuple(tensor, tensor) -- a tuple of upper and lower bound (x-\epsilon, x+\epsilon)
        """
        X = X.to(device)
        return (
            X.clamp(min=0) - self.epsilon,
            X.clamp(max=1) + self.epsilon,
        )

    def _indx_to_batch(self, batch_indx, start_indx, end_indx):
        """takes indices and returns batch input to the MIP solver
        
        Arguments:
            batch_indx {int} -- index of the batch in the data loader
            start_indx {int} -- start index of data point inside the selected batch
            end_indx {int} -- end endex of data point inside the selected batch
        
        Raises:
            ValueError: [description]
        
        Returns:
            tuple(tensor, np.array, tuple(tensor, tensor)) -- data points, labels and initial bounds for the MIP solver
        """
        for data_indx, data_itm in enumerate(self.data_loader):
            if data_indx == batch_indx:
                X, y = data_itm
                X, y = X.to(device), y.to(device)
                X, y = X[start_indx:end_indx], y[start_indx:end_indx]
                initial_bounds = self.get_initial_bounds(X)
                y = y.cpu().numpy()
                self._check_balanced(y)
                return X, y, initial_bounds
        raise ValueError(
            "[Exception] Batch Index {} start {} and end {} not correct out of {} available batches with {} elements per batch".format(
                batch_indx, start_indx, end_indx, self.n_batches, self.batch_size
            )
        )

    def _check_balanced(self, y):
        """sanity check for the balance of labels of currenly sampled batch
        
        Arguments:
            y {np.array} -- list of labels sampled for the MIP
        """
        if len(sorted(set(y))) != len(y):
            raise ValueError(
                "[Exception] Sampled Data Imbalance Error with labels {}".format(str(y))
            )

    def _get_mip_batch(self):
        """used toe get batch of data points for the MIP solver selected randomly/based on evaluator if exists
        Returns:
            tuple(tensor, np.array, tuple(tensor, tensor)) -- data points, labels and initial bounds for the MIP solver
        """
        self._increment_indices()
        batch_indx = self.batch_indx
        start_indx = self.start_indx
        end_indx = self.end_indx
        if self.model is not None and self.eval_function is not None:
            # includes a tuple of priority score and information of this batch ;)
            priority_score_list = []
            for cur_batch_indx, data_itm in enumerate(self.data_loader):
                X, y = data_itm
                X, y = X.to(device), y.to(device)
                for mip_batch_indx in range(self.n_mip_batches_in_loader_batch):
                    cur_start_indx = mip_batch_indx * self.config.num_samples
                    cur_end_indx = cur_start_indx + self.config.num_samples
                    if cur_end_indx >= len(y):
                        continue
                    priority_score, _ = self.eval_function(
                        self.model,
                        X[cur_start_indx:cur_end_indx],
                        y[cur_start_indx:cur_end_indx],
                    )
                    priority_score_list.append(
                        (priority_score, (cur_batch_indx, cur_start_indx, cur_end_indx))
                    )
            # ordering priority score
            priority_score_list = sorted(priority_score_list, key=lambda x: x[0])
            priority_indx = -1
            # random order
            if self.random_order:
                priority_indx = -1 * np.random.randint(
                    1, int(len(priority_score_list) * 0.05)
                )
            batch_indx, start_indx, end_indx = priority_score_list[priority_indx][1]
        X, y, initial_bounds = self._indx_to_batch(batch_indx, start_indx, end_indx)
        self.start_indx = self.end_indx
        self.end_indx = self.start_indx + self.config.num_samples
        if self.batch_indx >= self.n_batches:
            self._reset()
        return X, y, initial_bounds

    def get_n_output_classes(self):
        """used to get number of output classes in the current classification task
        
        Returns:
            int -- number of classes that needs to be predicted
        """
        current_dataset = self.data_loader.dataset
        if hasattr(current_dataset, "dataset"):
            current_dataset = current_dataset.dataset
        return len(current_dataset.class_to_idx.values())

    def get_input_n_channels(self):
        """used to compute input data point size and number of channels if possible
        
        Returns:
            tuple(tuple(w,h), int) -- returns input size and number of channels
        """
        X, _, _ = self._get_mip_batch()
        self._reset()
        if self.is_conv_model:
            # conv model
            input_size = X.shape[-2:]
        else:
            input_size = X.flatten().cpu().numpy().reshape(X.shape[0], -1).shape[-1]
        n_channels = X.shape[1]
        return input_size, n_channels

    def sample_from_class(self, label, n_images=1):
        """sampling from a specific class
        
        Arguments:
            label {int} -- label of the class that we will sample from
        
        Keyword Arguments:
            n_images {int} -- number of images to be sampled from that class (default: {1})
        
        Returns:
            tuple(tensor, np.array, tuple(tensor, tensor)) -- data points, labels and initial bounds for the MIP solver
        """
        data_points = []
        for _ in range(n_images):
            X, y = self.data_loader.dataset.sample_itm_class(label)
            data_points.append(X.unsqueeze(0))
            if y != label:
                raise ValueError(
                    "[Exception] Error in sampled label expected {} and got {}".format(
                        str(label), str(y)
                    )
                )
        X = torch.cat(data_points, 0).to(device)
        y = np.repeat(np.array([label]), n_images, axis=0)
        initial_bounds = self.get_initial_bounds(X)
        return X, y, initial_bounds

    def __iter__(self):
        """python iterator which resets index on creating an iterator from this object  
        
        Returns:
            MIPBatchLoader -- return current object iterator
        """
        self._reset()
        return self

    def __next__(self):
        """generate next batch in the list of possible batches from the input data loader for the MIp solver
        
        Returns:
            tuple(tensor, np.array, tuple(tensor, tensor)) -- data points, labels and initial bounds for the MIP solver
        """
        X, y, initial_bounds = self._get_mip_batch()
        return X, y, initial_bounds
