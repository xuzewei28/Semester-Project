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
Pin_memory = True

torch.manual_seed(0)
torch.cuda.manual_seed(0)
flag = False
data_set = 'cifar-100'
input_shape = [3, 32, 32]


def accuracy(loader, model, criterion, optimizer, device, name=''):
    """Compute the accuracy rate on the given dataset with the input model"""
    model.eval()
    log = dict()
    acc_loss = 0
    grad = 0
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

    log[name + '_loss'] = acc_loss / len(loader) / Batch_size
    log[name + "_grad"] = grad / len(loader)
    model.train()
    return log


def train_model(model, train_loader, test_loader, optimizer, criterion, T):
    """Train the model"""
    logs = []
    grad_logs = []
    names = list(n for n, _ in model.named_parameters())
    loop = tqdm(range(0, int(T)))
    count = 0
    for i in loop:
        loss = 0
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
        loop.set_postfix(i=i, loss=loss.item(), train_loss=log['train_loss'], train_grad=log['train_grad']
                         , test_loss=log['test_loss'], test_grad=log['test_grad'])

    return logs, grad_logs


def main(model, optimizer, criterion=nn.MSELoss(), name="AutoEncoder", path='Auto_Encoder_logs/AutoEncoder/'):
    print(name)
    model.to(device)

    dataset = CifarDataset(type="auto_encoder", data=data_set)
    train_loader = DataLoader(
        dataset,
        batch_size=Batch_size,
        shuffle=True,
        pin_memory=Pin_memory,
        num_workers=0
    )

    dataset = CifarDataset(train=False, type="auto_encoder", data=data_set)
    test_loader = DataLoader(
        dataset,
        batch_size=Batch_size,
        shuffle=True,
        pin_memory=Pin_memory,
        num_workers=0
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
    # torch.save(model, "model/LinearAutoEncoder/" + name + ".pth")


if __name__ == '__main__':
    """for i in range(2):
        flag = False
        name = 'SGD_lr_{0}_momentum_{1}'.format(0.1, 0)
        model = LinearAutoEncoder(input_shape=input_shape)
        optim = SGD(model.parameters(), lr=0.1, momentum=0)
        main(model, path="Cifar100/LinearAutoEncoder" + str(i) + "/", name=name,
             optimizer=optim)

        name = 'SGD_lr_{0}_momentum_{1}'.format(0.1, 0.9)
        model = LinearAutoEncoder(input_shape=input_shape)
        optim = SGD(model.parameters(), lr=0.1, momentum=0.9)
        main(model, path="Cifar100/LinearAutoEncoder" + str(i) + "/", name=name,
             optimizer=optim)

        name = 'Adam_lr_{0}'.format(1e-3)
        model = LinearAutoEncoder(input_shape=input_shape)
        optim = Adam(model.parameters(), lr=1e-3)
        main(model, path="Cifar100/LinearAutoEncoder" + str(i) + "/", name=name,
             optimizer=optim)

        name = 'Storm_lr_{0}'.format(1e-1)
        model = LinearAutoEncoder(input_shape=input_shape)
        optim = StormOptimizer(model.parameters(), lr=1e-1)
        main(model, path="Cifar100/LinearAutoEncoder" + str(i) + "/", name=name,
             optimizer=optim)

        flag = True
        name = 'SCRN_l_{0}_rho_{1}'.format(0.1, 0.5)
        model = LinearAutoEncoder(input_shape=input_shape)
        optim = SCRN(model.parameters(), l_=0.1, rho=0.5)
        main(model, path="Cifar100/LinearAutoEncoder" + str(i) + "/", name=name,
             optimizer=optim)

        name = 'HVP_RVR_SGD_lr_{0}_b_{1}_sigma2_{2}_l2_{3}'.format(0.1, 0.3, 0.1, 1)
        model = LinearAutoEncoder(input_shape=input_shape)
        optim = HVP_RVR(model.parameters(), lr=0.1, b=0.3, sigma2=0.1, l2=1)
        main(model, path="Cifar100/LinearAutoEncoder" + str(i) + "/", name=name,
             optimizer=optim)

    data_set = 'cifar-10'

    for i in range(2):
        flag = False
        name = 'SGD_lr_{0}_momentum_{1}'.format(0.1, 0)
        model = LinearAutoEncoder(input_shape=input_shape)
        optim = SGD(model.parameters(), lr=0.1, momentum=0)
        main(model, path="Cifar10/LinearAutoEncoder" + str(i) + "/", name=name,
             optimizer=optim)

        name = 'SGD_lr_{0}_momentum_{1}'.format(0.1, 0.9)
        model = LinearAutoEncoder(input_shape=input_shape)
        optim = SGD(model.parameters(), lr=0.1, momentum=0.9)
        main(model, path="Cifar10/LinearAutoEncoder" + str(i) + "/", name=name,
             optimizer=optim)

        name = 'Adam_lr_{0}'.format(1e-3)
        model = LinearAutoEncoder(input_shape=input_shape)
        optim = Adam(model.parameters(), lr=1e-3)
        main(model, path="Cifar10/LinearAutoEncoder" + str(i) + "/", name=name,
             optimizer=optim)

        name = 'Storm_lr_{0}'.format(1e-1)
        model = LinearAutoEncoder(input_shape=input_shape)
        optim = StormOptimizer(model.parameters(), lr=1e-1)
        main(model, path="Cifar10/LinearAutoEncoder" + str(i) + "/", name=name,
             optimizer=optim)

        flag = True
        name = 'SCRN_l_{0}_rho_{1}'.format(0.1, 0.5)
        model = LinearAutoEncoder(input_shape=input_shape)
        optim = SCRN(model.parameters(), l_=0.1, rho=0.5)
        main(model, path="Cifar10/LinearAutoEncoder" + str(i) + "/", name=name,
             optimizer=optim)

        name = 'HVP_RVR_SGD_lr_{0}_b_{1}_sigma2_{2}_l2_{3}'.format(0.1, 0.3, 0.1, 0.1)
        model = LinearAutoEncoder(input_shape=input_shape)
        optim = HVP_RVR(model.parameters(), lr=0.1, b=0.3, sigma2=0.1, l2=0.1)
        main(model, path="Cifar10/LinearAutoEncoder" + str(i) + "/", name=name,
             optimizer=optim)"""

    data_set = 'mnist'
    input_shape = [1, 28, 28]
    Num_epochs = 100

    flag = True
    name = 'SCRN_l_{0}_rho_{1}'.format(0.01, 0.05)
    model = LinearAutoEncoder(input_shape=input_shape)
    optim = SCRN(model.parameters(), l_=0.01, rho=0.05)
    main(model, path="MNIST/LinearAutoEncoder" + str(0) + "/", name=name,
         optimizer=optim)
    name = 'SCRN_l_{0}_rho_{1}'.format(0.01, 0.05)
    model = LinearAutoEncoder(input_shape=input_shape)
    optim = SCRN(model.parameters(), l_=0.01, rho=0.05)
    main(model, path="MNIST/LinearAutoEncoder" + str(1) + "/", name=name,
         optimizer=optim)

    name = 'SCRN_l_{0}_rho_{1}'.format(0.01, 0.05)
    model = LinearAutoEncoder(input_shape=input_shape)
    optim = SCRN(model.parameters(), l_=0.01, rho=0.05)
    main(model, path="MNIST/LinearAutoEncoder" + str(2) + "/", name=name,
         optimizer=optim)

    """for i in range(2):
        flag = False
        name = 'SGD_lr_{0}_momentum_{1}'.format(0.1, 0)
        model = LinearAutoEncoder(input_shape=input_shape)
        optim = SGD(model.parameters(), lr=0.1, momentum=0)
        main(model, path="MNIST/LinearAutoEncoder" + str(i) + "/", name=name,
             optimizer=optim)

        name = 'SGD_lr_{0}_momentum_{1}'.format(0.1, 0.9)
        model = LinearAutoEncoder(input_shape=input_shape)
        optim = SGD(model.parameters(), lr=0.1, momentum=0.9)
        main(model, path="MNIST/LinearAutoEncoder" + str(i) + "/", name=name,
             optimizer=optim)

        name = 'Adam_lr_{0}'.format(1e-3)
        model = LinearAutoEncoder(input_shape=input_shape)
        optim = Adam(model.parameters(), lr=1e-3)
        main(model, path="MNIST/LinearAutoEncoder" + str(i) + "/", name=name,
             optimizer=optim)

        name = 'Storm_lr_{0}'.format(1e-1)
        model = LinearAutoEncoder(input_shape=input_shape)
        optim = StormOptimizer(model.parameters(), lr=1e-1)
        main(model, path="MNIST/LinearAutoEncoder" + str(i) + "/", name=name,
             optimizer=optim)

        flag = True
        name = 'SCRN_l_{0}_rho_{1}'.format(0.1, 1)
        model = LinearAutoEncoder(input_shape=input_shape)
        optim = SCRN(model.parameters(), l_=0.1, rho=0.5)
        main(model, path="MNIST/LinearAutoEncoder" + str(i) + "/", name=name,
             optimizer=optim)

        name = 'HVP_RVR_SGD_lr_{0}_b_{1}_sigma2_{2}_l2_{3}'.format(0.1, 0.3, 0.1, 0.1)
        model = LinearAutoEncoder(input_shape=input_shape)
        optim = HVP_RVR(model.parameters(), lr=0.1, b=0.3, sigma2=0.1, l2=0.1)
        main(model, path="MNIST/LinearAutoEncoder" + str(i) + "/", name=name,
             optimizer=optim)"""
