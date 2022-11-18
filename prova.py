import torch
from torch.autograd import grad
import torch.nn as nn


# Create some dummy data.
x = torch.ones(2, 2, requires_grad=True)
gt = torch.ones_like(x) * 16 - 0.5  # "ground-truths"

# We will use MSELoss as an example.
loss_fn = nn.MSELoss()

# Do some computations.
v = x + 2
y = v ** 2

# Compute loss.
loss = loss_fn(y, gt)

print(f'Loss: {loss}')

# Now compute gradients:
d_loss_dx = grad(outputs=loss, inputs=x)
print(f'dloss/dx:\n {d_loss_dx}')