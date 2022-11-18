from torch import nn
from torch.autograd.functional import hessian, hvp
from torch.optim import Optimizer
import torch
from torch.nn.utils import stateless
from model import *
import numpy as np


class SCRN(Optimizer):
    def __init__(self, params, T_eps=10, l_=1,
                 rho=1, c_=1, eps=1e-9, device=None):
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
        self.log = []
        self.name = 'SCRN'

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
            temp = [g.reshape(-1) @ h.reshape(-1) / self.rho / a.pow(2) for g, h in zip(grad, hgp)]
            R_c = [(-t + torch.sqrt(t.pow(2) + 2 * a / self.rho)) for t in temp]
            delta = [-r * g / a for r, g in zip(R_c, grad)]
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

    def save_log(self, path='logs/logs/', flag_param=False):
        if flag_param:
            name = self.name + "_l_" + str(self.l) + "_rho_" + str(self.rho)
        else:
            name = self.name
        f = open(path + name, 'w')
        for l in self.log:
            f.write(str(l) + '\n')
        f.close()


class SCRN_Momentum(Optimizer):
    def __init__(self, params, momentum=0.9, T_eps=10, l_=1,
                 rho=1, c_=1, eps=1e-9, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        super(SCRN_Momentum, self).__init__(params, dict())
        self.hes = None
        self.T_eps = T_eps
        self.l_ = l_
        self.rho = rho
        self.c_ = c_
        self.eps = eps
        self.params = params
        self.f = None
        self.device = device
        self.log = []
        self.name = 'SCRN_Momentum'
        self.momentum = momentum
        self.old_delta = [torch.zeros(p.size()).to(device) for group in self.param_groups for p in group['params']]

    def set_f(self, f):
        self.f = f

    def step(self, **kwargs):
        grad = [p.grad for group in self.param_groups for p in group['params']]
        deltas = self.cubic_regularization(self.eps, grad)
        self.old_delta = [d1 * self.momentum + (1 - self.momentum) * d2 for d1, d2 in zip(self.old_delta, deltas)]
        for group in self.param_groups:
            for p, delta in zip(group["params"], self.old_delta):
                p.data += delta

    def cubic_regularization(self, eps, grad):
        g_norm = [torch.norm(g) for g in grad]
        a = sum(g_norm)
        if a >= ((self.l_ ** 2) / self.rho):
            hgp = hvp(self.f, tuple(p for group in self.param_groups for p in group['params']),
                      tuple(p.grad for group in self.param_groups for p in group['params']))[1]
            temp = [g.reshape(-1) @ h.reshape(-1) / self.rho / a.pow(2) for g, h in zip(grad, hgp)]
            R_c = [(-t + torch.sqrt(t.pow(2) + 2 * a / self.rho)) for t in temp]
            delta = [-r * g / a for r, g in zip(R_c, grad)]
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

    def save_log(self, path='logs/logs/', flag_param=False):
        if flag_param:
            name = self.name + "_l_" + str(self.l) + "_rho_" + str(self.rho)
        else:
            name = self.name
        f = open(path + name, 'w')
        for l in self.log:
            f.write(str(l) + '\n')
        f.close()


class SVRC(Optimizer):
    def __init__(self, params, l_=1,
                 rho=100, fp=1e-1, T_eps=10, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        super(SVRC, self).__init__(params, dict())
        self.hes = None
        self.l_ = l_
        self.rho = rho
        self.eps = min(self.l_ / (4 * self.rho), self.l_ ** 2 / (4 * self.rho))
        self.fp = fp
        self.Mt = 4 * self.rho
        self.T = 25 * self.rho ** 0.5 / self.eps ** 1.5
        # self.T_eps = int(T_eps*self.l_/(self.Mt*np.sqrt(self.eps/self.rho)))
        self.T_eps = T_eps
        self.params = params
        self.f = None
        self.device = device
        self.log = []
        self.name = 'SVRC'
        self.vt = None
        self.deltas = None

    def set_f(self, f):
        self.f = f

    def reset_vt(self):
        self.vt = None

    def step(self, **kwargs):
        if self.vt is None:
            self.vt = [p.grad for group in self.param_groups for p in group['params']]
        else:
            # bug not working
            p_grad = [p.grad for group in self.param_groups for p in group['params']]
            old_param = [p - d for p, d in
                         zip([p for group in self.param_groups for p in group['params']], self.deltas)]

            old_grad = torch.autograd.grad(self.f(*old_param), old_param)
            self.log.append((sum([torch.norm(o) for o in old_grad]), sum([torch.norm(o) for o in p_grad]), sum([torch.norm(o) for o in self.vt])))
            self.vt = [g1 - g2 + g3 for g1, g2, g3 in zip(p_grad, old_grad, self.vt)]
        self.deltas = self.cubic_regularization(self.vt, self.Mt, 1 / (16 * self.l_),
                                                np.sqrt(self.eps / self.rho), 0.5, self.fp / self.T / 3)
        for group in self.param_groups:
            for p, delta in zip(group["params"], self.deltas):
                p.data += delta

    def cubic_regularization(self, beta, tau, eta, zeta, eps, phi):
        n = sum([torch.norm(g) for g in beta])
        hgp = hvp(self.f, tuple(p for group in self.param_groups for p in group['params']),
                  tuple(beta))[1]
        temp = [g.reshape(-1) @ h.reshape(-1) / tau / n.pow(2) for g, h in zip(beta, hgp)]
        R_c = [(-t + torch.sqrt(t.pow(2) + 2 * n / tau)) for t in temp]
        x = [-r * g / n for r, g in zip(R_c, beta)]
        if self.cubic_function(beta, tau, x) <= -(1 - eps) * tau * (zeta ** 3) / 12:
            return x

        T_eps = self.T_eps
        sigma = (tau ** 2) * (zeta ** 3) * eps / (self.l_ + tau * zeta) / 576  # beta === rho?
        vec = [(torch.rand(g.size()) * 2 + torch.ones(g.size())).to(self.device) for g in beta]
        vec = [v / torch.norm(v) for v in vec]
        beta_ = [g + sigma * v for g, v in zip(beta, vec)]

        for i in range(T_eps):
            x = [a - eta * a1 for a, a1 in zip(x, self.cubic_grad(beta_, tau, x))]
            if self.cubic_function(beta_, tau, x) <= -(1 - eps) * tau * (zeta ** 3) / 12:
                return x
        return x

    def cubic_grad(self, beta, tau, x):
        hxp = hvp(self.f, tuple(p for group in self.param_groups for p in group['params']),
                  tuple(x))[1]
        nx = sum([torch.norm(a) for a in x])
        return [b + h + tau * nx * a / 2
                for a, b, h in zip(x, beta, hxp)]

    def cubic_function(self, beta, tau, x):
        hxp = hvp(self.f, tuple(p for group in self.param_groups for p in group['params']),
                  tuple(x))[1]
        nx = sum([torch.norm(a) for a in x])
        return sum([b.reshape(-1) @ a.reshape(-1) + a.reshape(-1) @ h.reshape(-1) / 2 + tau * (nx ** 3) / 6
                    for a, b, h in zip(x, beta, hxp)])

    def save_log(self, path='', flag_param=False, name=''):
        f = open(path + name, 'w')
        for l in self.log:
            f.write(str(l) + '\n')
        f.close()


class StormOptimizer(Optimizer):
    # Storing the parameters required in defaults dictionary
    # lr-->learning rate
    # c-->parameter to be swept over logarithmically spaced grid as per paper
    # w and k to be set as 0.1 as per paper
    # momentum-->dictionary storing model params as keys and their momentum term as values
    #            at each iteration(denoted by 'd' in paper)
    # gradient--> dictionary storing model params as keys and their gradients till now in a list as values
    #            (denoted by '∇f(x,ε)' in paper)
    # sqrgradnorm-->dictionary storing model params as keys and their sum of norm ofgradients till now
    #             as values(denoted by '∑G^2' in paper)

    def __init__(self, params, lr=0.1, c=100, momentum={}, gradient={}, sqrgradnorm={}):
        defaults = dict(lr=lr, c=c, momentum=momentum, sqrgradnorm=sqrgradnorm, gradient=gradient)
        super(StormOptimizer, self).__init__(params, defaults)

    # Returns the state of the optimizer as a dictionary containing state and param_groups as keys
    def __setstate__(self, state):
        super(StormOptimizer, self).__setstate__(state)

    # Performs a single optimization step for parameter updates
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        # param_groups-->a dict containing all parameter groups
        for group in self.param_groups:
            # Retrieving from defaults dictionary
            learn_rate = group['lr']
            factor = group['c']
            momentum = group['momentum']
            gradient = group['gradient']
            sqrgradnorm = group['sqrgradnorm']

            # Update step for each parameter present in param_groups
            for p in group['params']:
                # Calculating gradient('∇f(x,ε)' in paper)
                if p.grad is None:
                    continue
                dp = p.grad.data

                # Storing all gradients in a list
                if p in gradient:
                    gradient[p].append(dp)
                else:
                    gradient.update({p: [dp]})

                # Calculating and storing ∑G^2in sqrgradnorm
                if p in sqrgradnorm:
                    sqrgradnorm[p] = sqrgradnorm[p] + torch.pow(torch.norm(dp), 2)
                else:
                    sqrgradnorm.update({p: torch.pow(torch.norm(dp), 2)})

                # Updating learning rate('η' in paper)
                power = 1.0 / 3.0
                scaling = torch.pow((0.1 + sqrgradnorm[p]), power)
                learn_rate = learn_rate / (float)(scaling)

                # Calculating 'a' mentioned as a=cη^2 in paper(denoted 'c' as factor here)
                a = min(factor * learn_rate ** 2.0, 1.0)

                # Calculating and storing the momentum term(d'=∇f(x',ε')+(1-a')(d-∇f(x,ε')))
                if p in momentum:
                    momentum[p] = gradient[p][-1] + (1 - a) * (momentum[p] - gradient[p][-2])
                else:
                    momentum.update({p: dp})

                # Updation of model parameter p
                p.data = p.data - learn_rate * momentum[p]
                learn_rate = group['lr']

        return loss


if __name__ == '__main__':
    pass
