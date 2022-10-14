from torch import nn
from torch.autograd.functional import hessian, hvp
from torch.optim import Optimizer
import torch
from torch.nn.utils import stateless
from model import *


class SCRN(Optimizer):
    def __init__(self, params, T_eps=5, l_=1,
                 rho=100, c_=1, eps=1e-9, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        super(SCRN, self).__init__(params, dict())
        self.hes = None
        self.T_eps = T_eps
        self.l_ = l_
        self.rho = rho
        self.c_ = c_
        self.eps = eps
        self.params = params
        self.f = None
        self.device = device

    def set_f(self, f):
        self.f = f

    def step(self, **kwargs):
        grad = [p.grad for group in self.param_groups for p in group['params']]
        deltas = self.cubic_regularization(self.eps, grad)
        for group in self.param_groups:
            for p, delta in zip(group["params"], deltas):
                p.data += delta

    def cubic_regularization(self, eps, grad):
        g_norm = [torch.norm(g) for g in grad]
        hgp = hvp(self.f, tuple(p for group in self.param_groups for p in group['params']),
                  tuple(p.grad for group in self.param_groups for p in group['params']))[1]
        temp = [g.reshape(-1) @ h.reshape(-1) / self.rho / n.pow(2) for g, h, n in zip(grad, hgp, g_norm)]
        R_c = [(-t + torch.sqrt(t.pow(2) + 2 * n / self.rho)) for t, n in zip(temp, g_norm)]
        delta1 = [-r * g / n for r, g, n in zip(R_c, grad, g_norm)]
        delta2 = [torch.zeros(g.size()).to(self.device) for g in grad]
        sigma = self.c_ * (eps * self.rho) ** 0.5 / self.l_
        mu = 1.0 / (20.0 * self.l_)
        vec = [(torch.rand(g.size()) * 2 + torch.ones(g.size())).to(self.device) for g in grad]
        vec = [v / torch.norm(v) for v in vec]
        g_ = [g + sigma * v for g, v in zip(grad, vec)]
        for j in range(self.T_eps):
            hdp = hvp(self.f, tuple(p for group in self.param_groups for p in group['params']),
                      tuple(delta2))[1]
            delta2 = [(d - mu * (g + h + self.rho / 2 * torch.norm(d) * d)) for g, d, h in zip(g_, delta2, hdp)]
            #g_m = [(g + h + self.rho / 2 * torch.norm(d) * d) for g, d, h in zip(g_, delta2, hdp)]
            #d2_norm = [torch.norm(d) for d in g_m]
            
        delta = [(d1 if n >= ((self.l_ ** 2) / self.rho) else d2) for n, d1, d2 in zip(g_norm, delta1, delta2)]

        return delta


if __name__ == '__main__':
    model = vgg().cuda()
    criterion = nn.MSELoss()
    x = torch.rand(10, 3, 32, 32).cuda()
    y = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).cuda()
    y = torch.nn.functional.one_hot(y).float()
    names = list(n for n, _ in model.named_parameters())
    criterion(model(x), y).backward()


    def f(*params):
        out: torch.Tensor = stateless.functional_call(model, {n: p for n, p in zip(names, params)}, x)
        return criterion(out, y)


    hgp = hvp(f, tuple(p.data for p in model.parameters()), tuple(p.grad for p in model.parameters()))
    for h in hgp[1]:
        print(h.shape)
