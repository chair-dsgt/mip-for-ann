from torch.utils.data import Dataset
import numpy as np
import torch


class BalancedDataLoader(Dataset):
    """to sample a data point for each input class when we dont use shuffling used to sample balanced batches for MIP solver
    """

    def __init__(self, dataset, selected_indices=None):
        """initialize balanced data loader that loads balanced classes in each batch
        
        Arguments:
            dataset {torch.utils.data.Dataset} -- takes torch dataset as input
        
        Keyword Arguments:
            selected_indices {np.array} -- indices selected from input dataset if we want a subset of the input dataset (default: {None})
        """
        self.dataset = dataset
        self.targets = dataset.targets
        self.indices_per_class = {}
        self.selected_indices = selected_indices
        self.n_data = 0
        self.n_classes = 0
        self.map_index_indices = {}
        self.n_data_per_class = {}
        self._prepare_indices()

    def _prepare_indices(self):
        if type(self.targets) is list:
            self.targets = torch.tensor(self.targets)
        class_indxs = self.dataset.class_to_idx.values()
        labels = self.targets
        if self.selected_indices is not None:
            labels = labels[self.selected_indices]
        self.n_classes = len(class_indxs)
        for class_indx in class_indxs:
            class_indices = (
                torch.nonzero(torch.from_numpy(np.array(self.targets)) == class_indx, as_tuple=False)
                .numpy()
            )

            self.indices_per_class[class_indx] = class_indices
            self.n_data += len(class_indices)
            self.n_data_per_class[class_indx] = len(class_indices)

    def __len__(self):
        return self.n_data

    def __getitem__(self, index):
        class_indx = int(index % self.n_classes)
        current_indx = self.indices_per_class[class_indx][
            index % self.n_data_per_class[class_indx]
        ]
        X, y = self.dataset[current_indx.item()]
        return X, y

    def sample_rand_itm_class(self):
        """random sampling of a data point from a class
        
        Returns:
            tuple -- a data point along with its label
        """
        label = np.random.randint(0, self.n_classes)
        return self.sample_itm_class(label)

    def sample_itm_class(self, label):
        """samples a data point from a specific class
        
        Arguments:
            label {int} -- class index we wish to sample from
        
        Returns:
            tuple -- a data point along with its label
        """
        current_indx = self.indices_per_class[label][
            np.random.randint(0, self.n_data_per_class[label])
        ]
        return self.dataset[current_indx.item()]
