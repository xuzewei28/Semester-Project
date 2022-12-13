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
Pin_memory = True

torch.manual_seed(0)
torch.cuda.manual_seed(0)
flag = False


def accuracy(loader, model, criterion, device, name=''):
    """Compute the accuracy rate on the given dataset with the input model"""
    model.eval()
    log = dict()
    acc_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            loss = criterion(output, y)
            acc_loss += loss.item()
    log[name + '_loss'] = acc_loss / len(loader) / Batch_size
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
        train_log = accuracy(train_loader, model, criterion, device, name='train')
        test_log = accuracy(test_loader, model, criterion, device, name='test')
        log = {k: v for d in (train_log, test_log) for k, v in d.items()}
        logs.append(log)
        loop.set_postfix(i=i, loss=loss.item(), train_loss=log['train_loss'], test_loss=log['test_loss'])

    return logs, grad_logs


def main(model, optimizer, criterion=nn.MSELoss(), name="AutoEncoder", path='Auto_Encoder_logs/AutoEncoder/'):
    print(name)
    model.to(device)

    dataset = CifarDataset(type="auto_encoder")
    train_loader = DataLoader(
        dataset,
        batch_size=Batch_size,
        shuffle=True,
        pin_memory=Pin_memory,
        num_workers=0
    )

    dataset = CifarDataset(train=False, type="auto_encoder")
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
    flag = True
    # eps 1e-1 (simga1 1) l2 10 l1 1 simga2 1 ni 0.29
    for lr in [1e-2,1e-3,1e-4]:
        for b in [0.1, 0.3]:
            for sigma2 in [0.1, 1, 10, 20]:
                for l2 in [0.1, 1, 10, 20]:
                    for eps in [1e-2]:
                        name = 'lR_{0}_B_{1}_sigma_{2}_l2_{3}_eps_{4} '.format(lr, b, sigma2, l2, eps)
                        print(name)
                        model = LinearAutoEncoder()
                        optim = HVP_RVR(model.parameters(), lr=lr, b=b, sigma2=sigma2, l2=l2)
                        main(model, path="Auto_Encoder_logs/HVP_RVR/", name=name,
                             optimizer=optim)
