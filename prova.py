import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.models import VGG
from tqdm import tqdm


a=torch.zeros((10,2))
print(a.shape,(a*a).shape)