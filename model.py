import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F
from torchsummary import summary


class ResNet(nn.Module):
    def __init__(self, input_shape=None, n_class=10):
        super().__init__()
        if input_shape == None:
            input_shape = [3, 32, 32]
        self.res = torchvision.models.resnet18()
        self.res.fc = nn.Linear(512, n_class)

    def forward(self, x):
        return torch.sigmoid(self.res(x))


class OneLayerConvNet(nn.Module):
    def __init__(self, input_shape=None, n_class=10, n_hidden=32, act=None):
        super().__init__()
        if act is None:
            act = nn.ReLU()
        if input_shape == None:
            input_shape = [3, 32, 32]
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=n_hidden, kernel_size=3, stride=1,
                               padding=1,
                               bias=True)

        self.classifier = nn.Linear(n_hidden * 256, n_class, bias=True)
        self.act = act

    def forward(self, x):
        x = F.max_pool2d(self.act(self.conv1(x)), kernel_size=2).flatten(start_dim=1)

        return self.classifier(x)


class TwoLayerConvNet(nn.Module):
    def __init__(self, input_shape=None, n_class=10):
        super().__init__()
        if input_shape == None:
            input_shape = [3, 32, 32]
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.classifier = nn.Linear(4096, n_class, bias=True)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=2)
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2).flatten(start_dim=1)
        return torch.sigmoid(self.classifier(x))


class ThreeLayerConvNet(nn.Module):
    def __init__(self, input_shape=None, n_class=10):
        super().__init__()
        if input_shape == None:
            input_shape = [3, 32, 32]
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.classifier = nn.Linear(2048, n_class, bias=True)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=2)
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2)
        x = F.max_pool2d(F.relu(self.conv3(x)), kernel_size=2).flatten(start_dim=1)
        return torch.sigmoid(self.classifier(x))


class OneLayerLinearNet(nn.Module):
    def __init__(self, input_shape=None, n_class=10):
        super().__init__()
        if input_shape is None:
            input_shape = [3, 32, 32]
        self.classifier = nn.Linear(input_shape[0] * input_shape[1] * input_shape[2], n_class, bias=True)

    def forward(self, x):
        return self.classifier(x.flatten(start_dim=1))


class LinearNet(nn.Module):
    def __init__(self, input_shape=None, n_hidden=128, n_class=10):
        super().__init__()
        if input_shape is None:
            input_shape = [3, 32, 32]
        self.classifier = nn.Sequential(
            nn.Linear(input_shape[0] * input_shape[1] * input_shape[2], n_hidden, bias=True),
            nn.Linear(n_hidden, n_class))

    def forward(self, x):
        return torch.sigmoid(self.classifier(x.flatten(start_dim=1)))


class TwoLayerLinearNet(nn.Module):
    def __init__(self, input_shape=None, n_class=10, n_hidden=128):
        super().__init__()
        if input_shape is None:
            input_shape = [3, 32, 32]
        self.classifier = nn.Sequential(nn.Linear(input_shape[0] * input_shape[1] * input_shape[2], n_hidden),
                                        nn.ReLU(), nn.Linear(n_hidden, n_class))

    def forward(self, x):
        return torch.sigmoid(self.classifier(x.flatten(start_dim=1)))


class ThreeLayerLinearNet(nn.Module):
    def __init__(self, input_shape=None, n_class=10, n_hidden=128):
        super().__init__()
        if input_shape is None:
            input_shape = [3, 32, 32]
        self.classifier = nn.Sequential(nn.Linear(input_shape[0] * input_shape[1] * input_shape[2], n_hidden),
                                        nn.ReLU(), nn.Linear(n_hidden, n_hidden), nn.ReLU(),
                                        nn.Linear(n_hidden, n_class))

    def forward(self, x):
        return torch.sigmoid(self.classifier(x.flatten(start_dim=1)))


if __name__ == '__main__':
    net = ResNet(n_class=10, input_shape=(3, 32, 32))
    print(summary(net.cuda(), input_size=(3, 32, 32)))
