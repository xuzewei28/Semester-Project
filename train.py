import os
import torch.random
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import *
from optim import *
from dataset import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Learning_rate = 1e-4
Batch_size = 1  # 4
Num_epochs = 1e+6
Num_workers = 0
Pin_memory = True

torch.manual_seed(0)
torch.cuda.manual_seed(0)


def accuracy(loader, model, device, name=''):
    """Compute the accuracy rate on the given dataset with the input model"""
    model.eval()
    log = dict()
    num_correct = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            pred = torch.sigmoid(output).argmax(1)
            num_correct += (pred == y).item()

    log[name + '_acc'] = num_correct / len(loader) * 100
    model.train()
    return log


def train_model(model, train_loader, test_loader, optimizer, criterion, T):
    """Train the model"""
    logs = []
    for i in range(0, int(T), len(train_loader)):
        loop = tqdm(train_loader)
        for data, target in loop:
            data = data.to(device)
            target = target.to(device)
            with torch.cuda.amp.autocast():
                output = model(data)
                loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()

            optimizer.hessian(model, data)
            optimizer.step()
            train_log = accuracy(train_loader, model, device, name='train')
            test_log = accuracy(test_loader, model, device, name='test')
            log = {k: v for d in (train_log, test_log) for k, v in d.items()}
            logs.append(log)
            loop.set_postfix(i=i, loss=loss, train_acc=log['train_acc'], test_acc=log['test_acc'])

    return logs


def main():
    model = OneLayerNet()

    # Define the criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(),lr=1e-3)
    optimizer = SCRN(model.parameters())

    dataset = CifarDataset()
    train_loader = DataLoader(
        dataset,
        batch_size=Batch_size,
        shuffle=True,
        pin_memory=Pin_memory,
        num_workers=Num_workers
    )

    dataset = CifarDataset(train=False)
    test_loader = DataLoader(
        dataset,
        batch_size=Batch_size,
        shuffle=True,
        pin_memory=Pin_memory,
        num_workers=Num_workers
    )

    train_model(model, train_loader, test_loader, optimizer, criterion, Num_epochs)

    return net


if __name__ == '__main__':
    name = 'OneNetWithSGD'
    main()
