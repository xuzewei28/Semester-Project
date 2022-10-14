import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F
from torchsummary import summary


class OneLayerConvNet(nn.Module):
    def __init__(self, input_shape=None, n_class=10):
        super().__init__()
        if input_shape == None:
            input_shape = [3, 32, 32]
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=3, stride=1, padding=1,
                               bias=True)

        self.classifier = nn.Linear(8192, n_class, bias=True)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=2).flatten(start_dim=1)
        return torch.sigmoid(self.classifier(x))


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


class OneLayerLinearNet(nn.Module):
    def __init__(self, input_shape=None, n_class=10):
        super().__init__()
        if input_shape is None:
            input_shape = [3, 32, 32]
        self.classifier = nn.Linear(input_shape[0] * input_shape[1] * input_shape[2], n_class, bias=True)

    def forward(self, x):
        return torch.sigmoid(self.classifier(x.flatten(start_dim=1)))


class TwoLayerLinearNet(nn.Module):
    def __init__(self, input_shape=None, n_class=10):
        super().__init__()
        if input_shape is None:
            input_shape = [3, 32, 32]
        self.classifier = nn.Sequential(nn.Linear(input_shape[0] * input_shape[1] * input_shape[2], 1024),
                                        nn.ReLU(), nn.Linear(1024, n_class))

    def forward(self, x):
        return torch.sigmoid(self.classifier(x.flatten(start_dim=1)))


class vgg(nn.Module):
    def __init__(self, n_class=10, input_shape=(3, 32, 32)):
        super(vgg, self).__init__()
        model = torchvision.models.vgg11()  # False
        self.features = model.features
        for param in self.features.parameters():  # NOTE: prune:True  // finetune:False
            param.requires_grad = True

        temp = self.features(torch.rand(input_shape)).reshape(-1)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(temp.shape[0], 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, n_class)
        )

    def forward(self, x):
        return self.classifier(self.features(x).flatten(start_dim=1))


if __name__ == '__main__':
    net = TwoLayerLinearNet(n_class=10, input_shape=(3, 32, 32))
    print(summary(net.cuda(), input_size=(3, 32, 32)))
