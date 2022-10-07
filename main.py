
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 11:52:04 2022

@author: Saber Salehkaleybar
"""
import math
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np
from torch.autograd import Variable

# torch.manual_seed(2)


b = 0.2
rho = (4.603 - b) * 8
l_sgd = 2 + (4.603 - b) * 2
l = 2 + (4.603 - b) * 2
c_ = 1  # 1e-7

w_opt = 0
batch_size = 1
eps = 1e-6  # 1/(batch_size)
T_eps = 20  # int(l/(rho*eps)**0.5+1)
max_Instance = 1
Total_oracle_calls = 1  # 2500
sigma1 = 1
sigma2 = 1


def F(w):
    return torch.mean(torch.pow(w, 2) + (4.60333333333333 - b) * torch.pow(torch.sin(w), 2))


def f(w):
    return torch.mean((torch.pow(w, 2) + (4.60333333333333 - b) * torch.pow(torch.sin(w), 2)) + torch.normal(0, 2,
                                                                                                             size=(1,
                                                                                                                   w.size(
                                                                                                                       dim=1))))


def batch(w, batch_size):
    return w * torch.ones(batch_size)


def SGD(w_init, eps=None):
    eta = 10 ** -1
    w = Variable(w_init * torch.ones(1, 1), requires_grad=True)
    w_norm_history = []
    num_oracle_history = []
    f_history = []
    grad_history = []
    num_oracle = 0
    for i in range(int(Total_oracle_calls)):
        w_batch = batch(w, 1)
        F(w_batch).backward()
        grad = w.grad.clone()
        grad = grad + torch.normal(0, sigma1, size=(1, w.size(dim=1)))
        delta = (4 / (l_sgd)) * float(1 / (i + 1)) * grad
        w_new = w.detach() - delta
        w = w_new
        w_norm_history.append(torch.norm(w - w_opt).item())
        f_history.append(F(w).detach().numpy())
        grad_history.append(grad.detach().numpy().tolist()[0])
        num_oracle = num_oracle + batch_size
        num_oracle_history.append(num_oracle)
        w.requires_grad = True
    return w, w_norm_history, num_oracle_history, f_history


def cubic_regularization(eps, batch_size, w_init):
    w = Variable(w_init * torch.ones(2, 2), requires_grad=True)
    c = -(eps ** 3 / rho) ** (0.5) / 100
    w_norm_history = []
    num_oracle_history = []
    f_history = []
    grad_history = []
    num_oracle = 0

    w_batch = batch(w, batch_size)
    print(w_batch.shape)
    F(w_batch).backward()
    grad = w.grad.clone()[0]
    grad = grad + torch.normal(0, sigma1 / (batch_size) ** 0.5, size=(1, w.size(dim=1)))
    hessian = torch.autograd.functional.hessian(f, inputs=w_batch)  # [0][0][0]
    print(1, hessian, grad)




def cubic_subsolver(grad, hessian, eps):
    g_norm = torch.norm(grad)
    # print(g_norm)
    if g_norm > l ** 2 / rho:
        temp = grad @ hessian @ grad.T / rho / g_norm.pow(2)
        R_c = -temp + torch.sqrt(temp.pow(2) + 2 * g_norm / rho)
        delta = -R_c * grad / g_norm
    else:
        delta = torch.zeros(grad.size())
        sigma = c_ * (eps * rho) ** 0.5 / l
        mu = 1.0 / (20.0 * l)
        vec = torch.randn(grad.size())
        vec /= torch.norm(vec)
        g_ = grad + sigma * vec
        # g_ = grad
        for _ in range(T_eps):
            delta -= mu * (g_ + delta @ hessian + rho / 2 * torch.norm(delta) * delta)

    delta_m = grad @ delta.T + delta @ hessian @ delta.T / 2 + rho / 6 * torch.norm(delta).pow(3)
    return delta, delta_m


def cubic_finalsolver(grad, hessian, eps):
    delta = torch.zeros(grad.size())
    g_m = grad
    mu = 1 / (20 * l)
    while torch.norm(g_m) > eps / 2:
        delta -= mu * g_m
        g_m = grad + delta @ hessian + rho / 2 * torch.norm(delta) * delta
    return delta


# # # # ####cubic_regularization#######

total_hist_f = []
total_hist_f1 = []
total_hist_grad = []
for i in range(max_Instance):
    # SCRN
    w_init = 100 * np.random.uniform(-1, 1)
    cubic_regularization(eps, batch_size, w_init)

