import torch

a = torch.tensor([1, 0, 0], dtype=torch.float)
b = torch.ones(3, dtype=torch.float)
criterion = torch.nn.CrossEntropyLoss()

loss = criterion(a, b)
print(loss)
