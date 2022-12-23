import os

import numpy as np
import torch.random
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import *
from optim import *
from dataset import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Learning_rate = 1e-1
Batch_size = 500  # 4
Num_epochs = 300
Num_workers = 0
Pin_memory = True
one_hot = True
S = 10

torch.manual_seed(0)
torch.cuda.manual_seed(0)
flag = False
flag_svrc = False


class Regularization(nn.Module):
    def __init__(self, order=2, criterion=nn.MSELoss(), param=None, alpha=1e-5):
        super(Regularization, self).__init__()
        self.criterion = criterion
        self.order = order
        self.param = param
        self.alpha = alpha

    def forward(self, x, y):
        loss = self.criterion(x, y)
        if self.order == 1:
            return loss + self.alpha * sum(p.abs().sum() for p in self.param)
        elif self.order == 2:
            return loss + self.alpha * sum(p.pow(2).sum() for p in self.param)


def accuracy(loader, model, criterion, optimizer, device, name=''):
    """Compute the accuracy rate on the given dataset with the input model"""
    model.eval()
    log = dict()
    num_correct = 0
    grad = 0
    acc_loss = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        output = model(x)
        loss = criterion(output, y)
        acc_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        with torch.no_grad():
            grad += sum([torch.norm(p.grad).item() for p in model.parameters()])
            pred = torch.sigmoid(output).argmax(1)
            if one_hot:
                y = y.argmax(1)
            num_correct += (pred == y).sum().item()

    log[name + '_loss'] = acc_loss / len(loader)
    log[name + '_acc'] = num_correct / len(loader) / Batch_size * 100
    log[name + "_grad"] = grad / len(loader)
    model.train()
    optimizer.zero_grad()
    return log


def train_model(model, train_loader, test_loader, optimizer, criterion, T):
    """Train the model"""
    logs = []
    grad_logs = []
    names = list(n for n, _ in model.named_parameters())
    loop = tqdm(range(0, int(T)))
    count = 0

    for i in loop:
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            with torch.no_grad():
                grad = sum([torch.norm(p.grad).item() for p in model.parameters()])
                grad_logs.append(grad)

            def f(*params):
                out: torch.Tensor = stateless.functional_call(model, {n: p for n, p in zip(names, params)}, data)
                return criterion(out, target)

            if flag:
                optimizer.set_f(f)
            optimizer.step()
            count += 1
        train_log = accuracy(train_loader, model, criterion, optimizer, device, name='train')
        test_log = accuracy(test_loader, model, criterion, optimizer, device, name='test')
        log = {k: v for d in (train_log, test_log) for k, v in d.items()}
        logs.append(log)
        loop.set_postfix(i=i, loss=loss.item(),
                         train_acc=log['train_acc'], train_grad=log['train_grad'],
                         test_acc=log['test_acc'], test_grad=log['test_grad'], )

    return logs, grad_logs


def main(model, path, name, optimizer, criterion=nn.CrossEntropyLoss()):
    print(name)
    model.to(device)

    dataset = CifarDataset(one_hot=one_hot)
    train_loader = DataLoader(
        dataset,
        batch_size=Batch_size,
        shuffle=True,
        pin_memory=Pin_memory,
        num_workers=Num_workers
    )

    dataset = CifarDataset(train=False, one_hot=one_hot)
    test_loader = DataLoader(
        dataset,
        batch_size=Batch_size,
        shuffle=True,
        pin_memory=Pin_memory,
        num_workers=Num_workers
    )

    logs, grad_logs = train_model(model, train_loader, test_loader, optimizer, criterion, Num_epochs)
    os.makedirs(path, exist_ok=True)
    f = open(path + name, 'w')
    for l in logs:
        f.write(str(l) + '\n')
    f.close()

    os.makedirs("grad/" + path, exist_ok=True)
    f = open("grad/" + path + name, 'w')
    for l in grad_logs:
        f.write(str(l) + '\n')
    f.close()


if __name__ == '__main__':
    """   name = 'SGD_lR_{0}'.format(1e-2)
    model = OneLayerLinearNet()
    optim = SGD(model.parameters(), lr=1e-2)
    main(model, path="Final_logs/Classifier/OneLayerLinearNet/", name=name,
         optimizer=optim)

    name = 'SGD_lR_{0}_Momentum_{1}'.format(3e-3, 0.9)
    model = OneLayerLinearNet()
    optim = SGD(model.parameters(), lr=3e-3, momentum=0.9)
    main(model, path="Final_logs/Classifier/OneLayerLinearNet/", name=name,
         optimizer=optim)

    name = 'ADAM_lR_{0}'.format(1e-3)
    model = OneLayerLinearNet()
    optim = Adam(model.parameters(), lr=1e-3)
    main(model, path="Final_logs/Classifier/OneLayerLinearNet/", name=name,
         optimizer=optim)

    name = 'STORM_lR_{0}_c_{1}'.format(1e-1,100)
    model = OneLayerLinearNet()
    optim = StormOptimizer(model.parameters(), lr=1e-1, c=100)
    main(model, path="Final_logs/Classifier/OneLayerLinearNet/", name=name,
         optimizer=optim)

    flag = True
    name = 'SCRN_l_{0}_RHO_{1}'.format(1, 10)
    model = OneLayerLinearNet()
    optim = SCRN(model.parameters(), l_=1, rho=10)
    main(model, path="Final_logs/Classifier/OneLayerLinearNet/", name=name,
         optimizer=optim)

    name = 'HVP_RVR_SGD_lr_{0}_b_{1}_sigma2_{2}_l2_{3}'.format(1e-2, 0.3, 1, 0.1)
    model = OneLayerLinearNet()
    optim = HVP_RVR(model.parameters(), lr=0.01, b=0.3, sigma2=1, l2=0.1)
    main(model, path="Final_logs/Classifier/OneLayerLinearNet/", name=name,
         optimizer=optim)


    #ResNet
    flag=False
    name = 'SGD_lR_{0}'.format(1e-1)
    model = ResNet()
    optim = SGD(model.parameters(), lr=1e-1)
    main(model, path="Final_logs/Classifier/ResNet/", name=name,
         optimizer=optim)

    name = 'SGD_lR_{0}_Momentum_{1}'.format(3e-2, 0.9)
    model = ResNet()
    optim = SGD(model.parameters(), lr=3e-2, momentum=0.9)
    main(model, path="Final_logs/Classifier/ResNet/", name=name,
         optimizer=optim)

    name = 'ADAM_lR_{0}'.format(1e-3)
    model = ResNet()
    optim = Adam(model.parameters(), lr=1e-3)
    main(model, path="Final_logs/Classifier/ResNet/", name=name,
         optimizer=optim)

    name = 'STORM_lR_{0}_c_{1}'.format(1e-1, 100)
    model = ResNet()
    optim = StormOptimizer(model.parameters(), lr=1e-1, c=100)
    main(model, path="Final_logs/Classifier/ResNet/", name=name,
         optimizer=optim)

    flag = True
    name = 'SCRN_l_{0}_RHO_{1}'.format(0.1, 0.5)
    model = ResNet()
    optim = SCRN(model.parameters(), l_=0.1, rho=0.5)
    main(model, path="Final_logs/Classifier/ResNet/", name=name,
         optimizer=optim)

    name = 'HVP_RVR_SGD_lr_{0}_b_{1}_sigma2_{2}_l2_{3}'.format(1e-1, 0.1, 1, 10)
    model = ResNet()
    optim = HVP_RVR(model.parameters(), lr=0.1, b=0.1, sigma2=1, l2=10)
    main(model, path="Final_logs/Classifier/ResNet/", name=name,
         optimizer=optim)

    name = 'HVP_RVR_SGD_lr_{0}_b_{1}_sigma2_{2}_l2_{3}'.format(1e-2, 0.1, 1, 10)
    model = ResNet()
    optim = HVP_RVR(model.parameters(), lr=0.01, b=0.1, sigma2=1, l2=10)
    main(model, path="Final_logs/Classifier/ResNet/", name=name,
         optimizer=optim)"""

    """name = 'Adaptive_SGD_lR_{0}_t'.format(1e-2)
    model = OneLayerLinearNet()
    optim = Adaptive_SGD(model.parameters(), lr=1e-2, f=lambda x: x)
    main(model, path="Final_logs/Classifier/OneLayerLinearNet/", name=name,
         optimizer=optim)"""

    """name = 'Adaptive_SGD_lR_{0}_sqrt_t'.format(1e-2)
    model = OneLayerLinearNet()
    optim = Adaptive_SGD(model.parameters(), lr=1e-2, f=lambda x: np.sqrt(x))
    main(model, path="Final_logs/Classifier/OneLayerLinearNet/", name=name,
         optimizer=optim)

    name = 'Adaptive_SGD_lR_{0}_t'.format(1e-1)
    model = ResNet()
    optim = Adaptive_SGD(model.parameters(), lr=1e-1, f=lambda x: x)
    main(model, path="Final_logs/Classifier/ResNet/", name=name,
         optimizer=optim)

    name = 'Adaptive_SGD_lR_{0}_sqrt_t'.format(1e-1)
    model = ResNet()
    optim = Adaptive_SGD(model.parameters(), lr=1e-1, f=lambda x: np.sqrt(x))
    main(model, path="Final_logs/Classifier/ResNet/", name=name,
         optimizer=optim)"""
    flag = True
    for b in [0.3]:
        for sigma2 in [0.1, 1, 10]:
            for l1 in [0.1, 1, 10]:
                for l2 in [1, 5, 10]:
                    name = 'HVP_RVR_SCRN_b_{0}_sigma2_{1}_l1_{2}_l2_{3}'.format(b, sigma2, l1,
                                                                                l2 * l1)
                    model = OneLayerLinearNet()
                    optim = HVP_RVR(model.parameters(), b=b, sigma2=sigma2, l1=l1, l2=l2 * l1, mode='SCRN')
                    main(model, path="Prova/Classifier/OneLayerLinearNet/", name=name,
                         optimizer=optim)
