import os
import torch.random
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import *
from optim import *
from dataset import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Learning_rate = 1e-1
Batch_size = 1000  # 4
Num_epochs = 200
Num_workers = 0
Pin_memory = True

torch.manual_seed(0)
torch.cuda.manual_seed(0)
flag = False


def accuracy(loader, model, device, name=''):
    """Compute the accuracy rate on the given dataset with the input model"""
    model.eval()
    log = dict()
    num_correct = 0
    with torch.no_grad():
        for x, y in loader:
            # print(x.shape,y)
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            pred = torch.sigmoid(output).argmax(1)
            y = y.argmax(1)
            num_correct += (pred == y).sum().item()

    log[name + '_acc'] = num_correct / len(loader) / Batch_size * 100
    model.train()
    return log


def train_model(model, train_loader, test_loader, optimizer, criterion, T):
    """Train the model"""
    logs = []
    names = list(n for n, _ in model.named_parameters())
    loop = tqdm(range(0, int(T)))
    for i in loop:
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()

            def f(*params):
                out: torch.Tensor = stateless.functional_call(model, {n: p for n, p in zip(names, params)}, data)
                return criterion(out, target)

            if flag:
                optimizer.set_f(f)
            optimizer.step()
            train_log = accuracy(train_loader, model, device, name='train')
            test_log = accuracy(test_loader, model, device, name='test')
            log = {k: v for d in (train_log, test_log) for k, v in d.items()}
            logs.append(log)
            loop.set_postfix(i=i, loss=loss.item(), train_acc=log['train_acc'], test_acc=log['test_acc'])

    return logs


def main(model, name, optimizer):
    model.to(device)

    # Define the criterion and optimizer
    criterion = nn.MSELoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=Learning_rate)

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

    logs = train_model(model, train_loader, test_loader, optimizer, criterion, Num_epochs)
    f = open('logs/' + name, 'w')
    for l in logs:
        f.write(str(l) + '\n')
    f.close()


if __name__ == '__main__':
    models = [OneLayerLinearNet, OneLayerConvNet]
    names = ["OneLayerLinearNet", "OneLayerConvNet"]
    flag = True
    for model, name in zip(models, names):
        a = model()
        optimizer = SCRN(a.parameters())
        name1 = name + 'SCRN'
        print(name1)
        main(a, name1, optimizer)

    momentums = [ 0.9, 0]
    for model, name in zip(models, names):
        for momentum in momentums:
            a = model()
            optimizer = SGD(a.parameters(), lr=Learning_rate, momentum=momentum)
            name1 = name + "_SGD_lr_0.1_momentum_" + str(momentum)
            print(name1)
            main(a, name1, optimizer)
