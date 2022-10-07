from torch.autograd.functional import hessian
from torch.optim import Optimizer
import torch
from torch.nn.utils import stateless

class SCRN(Optimizer):
    def __init__(self, params, lr: float = 1e-3, momentum: float = 0.0, T_eps=20, l=1, rho=1, c_=1, eps=0):
        if lr <= 0.0:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if momentum < 0.0:
            raise ValueError('Invalid momentum value: {}'.format(momentum))

        defaults = dict(
            lr=lr,
            momentum=momentum,
        )
        super(SCRN, self).__init__(params, defaults)
        self.hes = None
        self.T_eps = T_eps
        self.l = l
        self.rho = rho
        self.c_ = c_
        self.eps = eps
        self.params = params

    def hessian(self, model, x):
        names = list(n for n, _ in model.named_parameters())
        def loss(*params):
            out: torch.Tensor = stateless.functional_call(model, {n: p for n, p in zip(names, params)}, x)
            return out.square().sum()

        self.hes = hessian(loss, tuple(model.parameters()))[0]

    def step(self, **kwargs):
        for group in self.param_groups:
            for p, hessian in zip(group["params"], self.hes):
                print(p, p.grad)
                delta = self.cubic_regularization(self.eps, p.grad, hessian)
                p.data += delta

    def cubic_regularization(self, eps, dw, hessian):
        grad = dw
        hessian = torch.sum(torch.reshape(hessian, (-1,))) # sum or avg
        delta, delta_m = self.cubic_subsolver(grad, hessian, eps)
        """if delta_m >= - 0.01 * ((eps ** 3) / self.rho) ** 0.5:
            delta = self.cubic_finalsolver(grad, hessian, eps)"""
        return delta

    def cubic_subsolver(self, grad, hessian, eps):
        g_norm = torch.norm(grad)
        # print(g_norm)
        if g_norm > self.l ** 2 / self.rho:

            temp = hessian * grad @ grad.T / self.rho / g_norm.pow(2)
            R_c = -temp + torch.sqrt(temp.pow(2) + 2 * g_norm / self.rho)
            print("----------------", grad.shape, R_c.shape, g_norm)
            if len(R_c.shape) == 0:
                delta = -R_c * grad / g_norm
            else:
                delta = -R_c @ grad / g_norm
        else:
            delta = torch.zeros(grad.size())
            sigma = self.c_ * (eps * self.rho) ** 0.5 / self.l
            mu = 1.0 / (20.0 * self.l)
            vec = torch.randn(grad.size())
            vec /= torch.norm(vec)
            g_ = grad + sigma * vec
            for _ in range(self.T_eps):
                delta -= mu * (g_ + delta * hessian + self.rho / 2 * torch.norm(delta) * delta)

        delta_m = grad @ delta.T + hessian * delta @ delta.T / 2 + self.rho / 6 * torch.norm(delta).pow(3)
        return delta, delta_m

    def cubic_finalsolver(self, grad, hessian, eps):
        delta = torch.zeros(grad.size())
        g_m = grad
        mu = 1 / (20 * self.l)
        while torch.norm(g_m) > eps / 2:
            delta -= mu * g_m
            g_m = grad + delta * hessian + self.rho / 2 * torch.norm(delta) * delta
        return delta


if __name__ == '__main__':
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 3, kernel_size=3, bias=True))
    scrn = SCRN(model.parameters())
    x = torch.rand(1, 3, 16, 16)
    loss = model(x).square().sum()
    loss.backward()
    scrn.hessian(model, x)
    scrn.step()
    print("end")
