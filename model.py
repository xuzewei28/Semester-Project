import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F
from torchvision.transforms.functional import resize
from torchsummary import summary
import torchvision.models as model


class OneLayerNet(nn.Module):
    def __init__(self, input_shape=None, n_class=10):
        super().__init__()
        if input_shape == None:
            input_shape = [3, 32, 32]
        self.conv = nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=3, stride=2)
        temp = F.max_pool2d(self.conv.forward(torch.zeros(size=input_shape)), kernel_size=2).flatten()
        print(temp.shape)
        self.classifier = nn.Linear(temp.shape[0], n_class)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv(x)), kernel_size=2).flatten(start_dim=1)
        return self.classifier(x)


if __name__ == '__main__':
    net = OneLayerNet()
    summary(net, input_size=(3, 32, 32))
