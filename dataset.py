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
            train_data, train_labels, test_data, test_labels = \
                self.load_cifar_10_data()
        elif data == 'mnist':
            train_data, train_labels, test_data, test_labels = \
                self.load_mnist_data()
        elif data == 'cifar-100':
            train_data, train_labels, test_data, test_labels = \
                self.load_cifar_100_data()

        else:
            train_data, train_labels, test_data, test_labels = None, None, None, None

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

        train_input = mnist_train_set.data.view(-1, 28, 28, 1).float()
        train_target = mnist_train_set.targets
        test_input = mnist_test_set.data.view(-1, 28, 28, 1).float()
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

    def load_cifar_10_data(self):
        train_set = datasets.CIFAR10('./data/cifar10/', train=True, download=True)
        test_set = datasets.CIFAR10('./data/cifar10/', train=False, download=True)

        train_input = train_set.data
        train_target = train_set.targets
        test_input = test_set.data
        test_target = test_set.targets

        return train_input, train_target, test_input, test_target

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

    def get_item_with_label(self, label):
        index = self.labels[self.labels[:, label] == 1]
        pos = np.random.randint(0, len(index), 1)

        return self.data[self.labels[:, label] == 1][pos][0]


if __name__ == '__main__':
    df = CifarDataset(data='cifar-10')
    images = []
    for i in range(30):
        images.append(df.get_item_with_label(i % 10))


    def show_image(images):
        from PIL import Image
        output = np.zeros((96, 320, 3))
        for i in range(30):
            im = images[i]
            im = im.transpose(0, -1).transpose(0, 1).numpy()
            output[i // 10 * 32:(i // 10 + 1) * 32, i % 10 * 32:(i % 10 + 1) * 32] = im

        return Image.fromarray(output.astype(np.uint8), mode='RGB')


    im = show_image(images)
    im.save('Cifar10_image.png')
