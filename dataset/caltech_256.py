from __future__ import print_function
from PIL import Image
import os
import os.path
import glob

from matplotlib.pyplot import imread
import torch
import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity

# https://authors.library.caltech.edu/7694/
# Adapted from https://gist.github.com/yangarbiter/33a706011d1a833485fdc5000df55d25


class Caltech256(data.Dataset):
    """`Caltech256.
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

    base_folder = "256_ObjectCategories"
    url = "http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar"
    filename = "256_ObjectCategories.tar"
    tgz_md5 = "67b4f42ca05d46448c6bb8ecd2220f6d"

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
        self.class_to_idx = {}
        self.data = []
        self.labels = []
        for cat in range(1, 258):
            cat_dirs = glob.glob(
                os.path.join(self.root, self.base_folder, "%03d*" % cat)
            )
            for fdir in cat_dirs:
                for fimg in glob.glob(os.path.join(fdir, "*.jpg")):
                    # img = imread(fimg)
                    # img = Image.open(fimg).convert("RGB")

                    self.data.append(fimg)
                    self.labels.append(cat - 1)
                    self.class_to_idx[cat] = cat - 1

                # self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
        self.targets = torch.tensor(self.labels)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = imread(img)
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        fpath = os.path.join(self.root, self.filename)
        if not check_integrity(fpath, self.tgz_md5):
            return False
        return True

    def download(self):
        import tarfile

        root = self.root
        download_url(self.url, root, self.filename, self.tgz_md5)

        # extract file
        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(root, self.filename), "r")
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
