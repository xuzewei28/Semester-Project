from typing import List, Union, Any

from torch import nn
from torch.autograd.functional import hessian, hvp
from torch.optim import Optimizer
import torch
from torch.nn.utils import stateless
from model import *


class SCRN(Optimizer):
    def __init__(self, params, T_eps=10, l_=1,
                 rho=10, c_=1, eps=1e-9, device=None, flag_adam=False, lr=1e-3, b1=0.9, b2=0.999):
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
        self.flag_adam = flag_adam
        self.b1 = b1
        self.b2 = b2
        self.lr = lr
        self.mt = [torch.zeros(p.size()).to(device) for group in self.param_groups for p in group['params']]
        self.vt = [torch.zeros(p.size()).to(device) for group in self.param_groups for p in group['params']]
        self.log = []
        self.name = 'SCRN_l_' + str(l_) + "_rho_" + str(rho)

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
        a = sum(g_norm)
        if a >= ((self.l_ ** 2) / self.rho):

            hgp = hvp(self.f, tuple(p for group in self.param_groups for p in group['params']),
                      tuple(p.grad for group in self.param_groups for p in group['params']))[1]
            temp = [g.reshape(-1) @ h.reshape(-1) / self.rho / n.pow(2) for g, h, n in zip(grad, hgp, g_norm)]
            R_c = [(-t + torch.sqrt(t.pow(2) + 2 * n / self.rho)) for t, n in zip(temp, g_norm)]
            delta = [-r * g / n for r, g, n in zip(R_c, grad, g_norm)]
            self.log.append(('1', a.item(), sum([torch.norm(d) for d in delta]).item()))
        else:
            delta = [torch.zeros(g.size()).to(self.device) for g in grad]
            sigma = self.c_ * (eps * self.rho) ** 0.5 / self.l_
            mu = 1.0 / (20.0 * self.l_)
            vec = [(torch.rand(g.size()) * 2 + torch.ones(g.size())).to(self.device) for g in grad]
            vec = [v / torch.norm(v) for v in vec]
            g_ = [g + sigma * v for g, v in zip(grad, vec)]
            for j in range(self.T_eps):
                hdp = hvp(self.f, tuple(p for group in self.param_groups for p in group['params']),
                          tuple(delta))[1]
                delta = [(d - mu * (g + h + self.rho / 2 * torch.norm(d) * d)) for g, d, h in zip(g_, delta, hdp)]
                # g_m = [(g + h + self.rho / 2 * torch.norm(d) * d) for g, d, h in zip(g_, delta2, hdp)]
                # d2_norm = [torch.norm(d) for d in g_m]
            self.log.append(('2', a.item(), sum([torch.norm(d) for d in delta]).item()))
        return delta

    def adam(self, delta):
        self.mt = [self.b1 * m + (1 - self.b1) * g for m, g in zip(self.mt, delta)]
        self.vt = [self.b2 * v + (1 - self.b2) * (g * g) for v, g in zip(self.vt, delta)]
        mt = [m / (1 - self.b1) for m in self.mt]
        vt = [v / (1 - self.b2) for v in self.vt]
        return [self.lr * m / (torch.sqrt(v) + 1e-8) for m, v in zip(mt, vt)]

    def save_log(self):
        f = open('logs/logs/' + self.name, 'w')
        for l in self.log:
            f.write(str(l) + '\n')
        f.close()



class SCRN1(Optimizer):
    def __init__(self, params, T_eps=10, l_=1,
                 rho=10, c_=1, eps=1e-9, device=None, flag_adam=False, lr=1e-3, b1=0.9, b2=0.999):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        super(SCRN1, self).__init__(params, dict())
        self.hes = None
        self.T_eps = T_eps
        self.l_ = l_
        self.rho = rho
        self.c_ = c_
        self.eps = eps
        self.params = params
        self.f = None
        self.device = device
        self.flag_adam = flag_adam
        self.b1 = b1
        self.b2 = b2
        self.lr = lr
        self.mt = [torch.zeros(p.size()).to(device) for group in self.param_groups for p in group['params']]
        self.vt = [torch.zeros(p.size()).to(device) for group in self.param_groups for p in group['params']]
        self.log = []
        self.name = 'SCRN1_l_' + str(l_) + "_rho_" + str(rho)

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
        a = sum(g_norm)
        if a >= ((self.l_ ** 2) / self.rho):

            hgp = hvp(self.f, tuple(p for group in self.param_groups for p in group['params']),
                      tuple(p.grad for group in self.param_groups for p in group['params']))[1]
            temp = [g.reshape(-1) @ h.reshape(-1) / self.rho / a.pow(2) for g, h, n in zip(grad, hgp, g_norm)]
            R_c = [(-t + torch.sqrt(t.pow(2) + 2 * a / self.rho)) for t, n in zip(temp, g_norm)]
            delta = [-r * g / a for r, g, n in zip(R_c, grad, g_norm)]
            self.log.append(('1', a.item(), sum([torch.norm(d) for d in delta]).item()))
        else:
            delta = [torch.zeros(g.size()).to(self.device) for g in grad]
            sigma = self.c_ * (eps * self.rho) ** 0.5 / self.l_
            mu = 1.0 / (20.0 * self.l_)
            vec = [(torch.rand(g.size()) * 2 + torch.ones(g.size())).to(self.device) for g in grad]
            vec = [v / torch.norm(v) for v in vec]
            g_ = [g + sigma * v for g, v in zip(grad, vec)]
            for j in range(self.T_eps):
                hdp = hvp(self.f, tuple(p for group in self.param_groups for p in group['params']),
                          tuple(delta))[1]
                delta = [(d - mu * (g + h + self.rho / 2 * torch.norm(d) * d)) for g, d, h in zip(g_, delta, hdp)]
                # g_m = [(g + h + self.rho / 2 * torch.norm(d) * d) for g, d, h in zip(g_, delta2, hdp)]
                # d2_norm = [torch.norm(d) for d in g_m]
            self.log.append(('2', a.item(), sum([torch.norm(d) for d in delta]).item()))
        return delta

    def adam(self, delta):
        self.mt = [self.b1 * m + (1 - self.b1) * g for m, g in zip(self.mt, delta)]
        self.vt = [self.b2 * v + (1 - self.b2) * (g * g) for v, g in zip(self.vt, delta)]
        mt = [m / (1 - self.b1) for m in self.mt]
        vt = [v / (1 - self.b2) for v in self.vt]
        return [self.lr * m / (torch.sqrt(v) + 1e-8) for m, v in zip(mt, vt)]

    def save_log(self):
        f = open('logs/logs/' + self.name, 'w')
        for l in self.log:
            f.write(str(l) + '\n')
        f.close()




if __name__ == '__main__':
    pass
