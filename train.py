import os
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
Num_epochs = 100
Num_workers = 0
Pin_memory = True
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
            y=y.argmax(1)
            num_correct += (pred == y).sum().item()

    log[name + '_acc'] = num_correct / len(loader) / Batch_size * 100
    model.train()
    return log


def train_model(model, train_loader, test_loader, optimizer, criterion, T):
    """Train the model"""
    logs = []
    names = list(n for n, _ in model.named_parameters())
    loop = tqdm(range(0, int(T)))
    count = 0
    data1 = []
    target1 = []
    for i in loop:
        for data, target in train_loader:
            if flag_svrc:
                if count % S == 0:
                    data1 = []
                    target1 = []
                    count = 1
                    optimizer.reset_vt()
                if len(data1) < S:
                    data1.append(data)
                    target1.append(target)
                    if len(data1) == S:
                        data = torch.cat(tuple(data1))
                        target = torch.cat(tuple(target1))
                    else:
                        continue
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
            count += 1
        train_log = accuracy(train_loader, model, device, name='train')
        test_log = accuracy(test_loader, model, device, name='test')
        log = {k: v for d in (train_log, test_log) for k, v in d.items()}
        logs.append(log)
        loop.set_postfix(i=i, loss=loss.item(), train_acc=log['train_acc'], test_acc=log['test_acc'])

    return logs


def main(model, name, optimizer, criterion):
    model.to(device)

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
    flag = True
    print("SCRN+L2+MSELoss+ResNet")
    for l in [1]:
        for rho in [10]:
            for alpha in [0]:
                name = "SCRN_l_" + str(l) + "_rho_" + str(l * rho)
                print(name)
                model = ResNet()
                criterion = Regularization(param=model.parameters(), alpha=alpha)
                optim = SCRN(model.parameters(), l_=l, rho=l * rho)
                main(model, "MSELoss/ResNet/" + name, criterion=criterion, optimizer=optim)
