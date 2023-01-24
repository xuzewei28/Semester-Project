import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F
from torchsummary import summary


class LinearEncoder(nn.Module):
    def __init__(self, input_shape=None):
        super().__init__()
        if input_shape is None:
            input_shape = [3, 32, 32]
        self.l1 = nn.Linear(input_shape[0] * input_shape[1] * input_shape[2], 1000)
        self.l2 = nn.Linear(1000, 500)
        self.l3 = nn.Linear(500, 250)
        self.l4 = nn.Linear(250, 30)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return x


class LinearDecoder(nn.Module):
    def __init__(self, output_shape=None):
        super().__init__()
        if output_shape is None:
            input_shape = [3, 32, 32]
        self.l1 = nn.Linear(30, 250)
        self.l2 = nn.Linear(250, 500)
        self.l3 = nn.Linear(500, 1000)
        self.l4 = nn.Linear(1000, output_shape[0] * output_shape[1] * output_shape[2])
        self.shape = output_shape

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.l4(x).reshape([x.shape[0], self.shape[0], self.shape[1], self.shape[2]])
        return x


class LinearAutoEncoder(nn.Module):
    def __init__(self, input_shape=None):
        super().__init__()
        if input_shape is None:
            input_shape = [3, 32, 32]
        self.encoder = LinearEncoder(input_shape)
        self.decoder = LinearDecoder(input_shape)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class Encoder(nn.Module):
    def __init__(self, input_shape=None):
        super().__init__()
        if input_shape is None:
            input_shape = [3, 32, 32]
        self.l1 = nn.Conv2d(input_shape[0], 32, kernel_size=5, stride=1)
        self.l2 = nn.Conv2d(32, 32, kernel_size=5, stride=1)
        self.l3 = nn.Conv2d(32, 32, kernel_size=4, stride=2)
        self.l4 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.l5 = nn.Conv2d(32, 8, kernel_size=4, stride=1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)


class Decoder(nn.Module):
    def __init__(self, output_shape=None):
        super().__init__()
        if output_shape is None:
            input_shape = [3, 32, 32]
        self.l1 = nn.ConvTranspose2d(8, 32, kernel_size=4, stride=1)
        self.l2 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2)
        self.l3 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2)
        self.l4 = nn.ConvTranspose2d(32, 32, kernel_size=5, stride=1)
        self.l5 = nn.ConvTranspose2d(32, output_shape[0], kernel_size=5, stride=1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = self.l5(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, input_shape=None):
        super().__init__()
        if input_shape is None:
            input_shape = [3, 32, 32]
        self.encoder = Encoder(input_shape)
        self.decoder = Decoder(input_shape)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_shape=None):
        super().__init__()
        if input_shape is None:
            input_shape = [3, 32, 32]
        self.encoder = Encoder(input_shape)
        self.decoder = Decoder(input_shape)

    def forward(self, x):
        param_f = self.encoder(x)
        mu_f, logvar_f = param_f.split(param_f.size(1) // 2, 1)
        std_f = torch.exp(0.5 * logvar_f)
        z = torch.randn_like(mu_f) * std_f + mu_f
        return self.decoder(z)




class ResNet(nn.Module):
    def __init__(self, input_shape=None, n_class=10):
        super().__init__()
        self.input_layer = None
        if input_shape == None:
            input_shape = [3, 32, 32]
        if input_shape[0] != 3:
            self.input_layer = lambda x: torch.cat([x, x, x], dim=1)
        self.res = torchvision.models.resnet18()
        self.res.fc = nn.Linear(512, n_class)

    def forward(self, x):
        if self.input_layer is not None:
            x=self.input_layer(x)

        return self.res(x)


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
    net = AutoEncoder()
    print(summary(net.cuda(), input_size=(3, 32, 32)))
