from __future__ import print_function
from PIL import Image
import os
import os.path
import glob

from matplotlib.pyplot import imread
import torch
import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity
import torchvision.datasets as datasets

# Adapted from https://github.com/tjmoon0104/Tiny-ImageNet-Classifier/blob/master/ResNet18_Baseline.ipynb
# https://tiny-imagenet.herokuapp.com/


class TinyImageNet(data.Dataset):
    """`TinyImageNet.
    Args:
        root (string): Root directory of dataset where directory
            ``256_ObjectCategories`` exists.
        train (bool, optional): Not used
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    base_folder = "tiny-imagenet-200"
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    filename = "tiny-imagenet-200.zip"
    tgz_md5 = "90528d7ca1a48142e341f4ef8d21d0de"

    def __init__(
        self, root, train=True, transform=None, target_transform=None, download=False
    ):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "[Exception] Dataset not found or corrupted."
                + " You can use download=True to download it"
            )
        if train:
            self.dataset = datasets.ImageFolder(
                os.path.join(self.root, self.base_folder, "train"),
                transform=transform,
                target_transform=target_transform,
            )
        else:
            self.dataset = datasets.ImageFolder(
                os.path.join(self.root, self.base_folder, "test"),
                transform=transform,
                target_transform=target_transform,
            )
        self.class_to_idx = self.dataset.class_to_idx
        self.labels = self.dataset.targets
        self.targets = torch.tensor(self.labels)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def _check_integrity(self):
        fpath = os.path.join(self.root, self.filename)
        if not check_integrity(fpath, self.tgz_md5):
            return False
        return True

    def download(self):
        from zipfile import ZipFile

        root = self.root
        download_url(self.url, root, self.filename, self.tgz_md5)

        # extract file
        cwd = os.getcwd()
        tar = ZipFile(os.path.join(root, self.filename), "r")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)

    def __repr__(self):
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += "    Number of datapoints: {}\n".format(self.__len__())
        fmt_str += "    Root Location: {}\n".format(self.root)
        tmp = "    Transforms (if any): "
        fmt_str += "{0}{1}\n".format(
            tmp, self.transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        tmp = "    Target Transforms (if any): "
        fmt_str += "{0}{1}".format(
            tmp, self.target_transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        return fmt_str
