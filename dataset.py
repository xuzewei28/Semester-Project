import torch
from torch.utils.data import Dataset
import numpy as np
import pickle

from torchvision import datasets


class CifarDataset(Dataset):
    """The class RoadDataset loads the data and executes the pre-processing operations on it"""

    def __init__(self, path='data/cifar-10-batches-py', train=True, one_hot=True, len_data=None, type='classifier',
                 data='cifar-10'):
        if data == 'cifar-10':
            train_data, train_filenames, train_labels, test_data, test_filenames, test_labels, label_names = \
                self.load_cifar_10_data(path)
        elif data == 'mnist':
            train_data, train_labels, test_data, test_labels = \
                self.load_mnist_data()
        elif data=='cifar-100':
            train_data, train_labels, test_data, test_labels = \
                self.load_cifar_100_data()

        else:
            train_data, train_labels, test_data, test_labels=None,None,None,None

        if train:
            self.data = train_data
            self.labels = train_labels
        else:
            self.data = test_data
            self.labels = test_labels

        self.data = torch.tensor(self.data).transpose(1, -1).transpose(2, -1)
        self.labels = torch.tensor(self.labels).long()

        if one_hot:
            self.labels = torch.nn.functional.one_hot(self.labels).float()

        if len_data is None:
            self.len_data = len(self.data)
        else:
            self.len_data = len_data

        self.type = type

    def unpickle(self, file):
        """load the cifar-10 data"""
        with open(file, 'rb') as fo:
            data = pickle.load(fo, encoding='bytes')
        return data

    def load_mnist_data(self):
        mnist_train_set = datasets.MNIST('./data/mnist/', train=True, download=True)
        mnist_test_set = datasets.MNIST('./data/mnist/', train=False, download=True)

        train_input = mnist_train_set.data.view(-1, 28, 28,1).float()
        train_target = mnist_train_set.targets
        test_input = mnist_test_set.data.view(-1, 28, 28,1).float()
        test_target = mnist_test_set.targets

        return train_input, train_target, test_input, test_target

    def load_cifar_100_data(self):
        train_set = datasets.CIFAR100('./data/cifar100/', train=True)
        test_set = datasets.CIFAR100('./data/cifar100/', train=False)

        train_input = train_set.data
        train_target = train_set.targets
        test_input = test_set.data
        test_target = test_set.targets

        return train_input, train_target, test_input, test_target

    def load_cifar_10_data(self, data_dir, negatives=False):
        """
        Return train_data, train_filenames, train_labels, test_data, test_filenames, test_labels
        """

        meta_data_dict = self.unpickle(data_dir + "/batches.meta")
        cifar_label_names = meta_data_dict[b'label_names']
        cifar_label_names = np.array(cifar_label_names)

        # training data
        cifar_train_data = None
        cifar_train_filenames = []
        cifar_train_labels = []

        for i in range(1, 6):
            cifar_train_data_dict = self.unpickle(data_dir + "/data_batch_{}".format(i))
            if i == 1:
                cifar_train_data = cifar_train_data_dict[b'data']
            else:
                cifar_train_data = np.vstack((cifar_train_data, cifar_train_data_dict[b'data']))
            cifar_train_filenames += cifar_train_data_dict[b'filenames']
            cifar_train_labels += cifar_train_data_dict[b'labels']

        cifar_train_data = cifar_train_data.reshape((len(cifar_train_data), 3, 32, 32))
        if negatives:
            cifar_train_data = cifar_train_data.transpose(0, 2, 3, 1).astype(np.float32)
        else:
            cifar_train_data = np.rollaxis(cifar_train_data, 1, 4)
        cifar_train_filenames = np.array(cifar_train_filenames)
        cifar_train_labels = np.array(cifar_train_labels)

        # test data
        # cifar_test_data_dict
        # 'batch_label': 'testing batch 1 of 1'
        # 'data': ndarray
        # 'filenames': list
        # 'labels': list

        cifar_test_data_dict = self.unpickle(data_dir + "/test_batch")
        cifar_test_data = cifar_test_data_dict[b'data']
        cifar_test_filenames = cifar_test_data_dict[b'filenames']
        cifar_test_labels = cifar_test_data_dict[b'labels']

        cifar_test_data = cifar_test_data.reshape((len(cifar_test_data), 3, 32, 32))
        if negatives:
            cifar_test_data = cifar_test_data.transpose(0, 2, 3, 1).astype(np.float32)
        else:
            cifar_test_data = np.rollaxis(cifar_test_data, 1, 4)
        cifar_test_filenames = np.array(cifar_test_filenames)
        cifar_test_labels = np.array(cifar_test_labels)

        return cifar_train_data, cifar_train_filenames, cifar_train_labels, \
               cifar_test_data, cifar_test_filenames, cifar_test_labels, cifar_label_names

    def __len__(self):
        return self.len_data

    def __getitem__(self, index):
        """This method returns the image at a certain position and its mask"""
        image = self.data[index]
        label = self.labels[index]
        if self.type == "auto_encoder":
            return (image / 255), (image / 255)
        if self.type == 'classifier':
            return (image / 255), label


if __name__ == '__main__':
    df = CifarDataset(data='cifar-100')
    x,y=df.__getitem__(1)
    print(x.shape,y)
