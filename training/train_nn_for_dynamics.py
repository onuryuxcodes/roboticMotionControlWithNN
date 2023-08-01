import numpy as np
import pandas as pd
from training.log_messages import print_loss
from dynamics.constants import column_list
import torch


def train(nn_lyapunov, nn_policy, t, e, e_and_t, zeros_and_t, f_of_e, b_friction_constant, alpha, max_iterations,
          optimizer_l,
          optimizer_p):
    n = len(e)
    loss_each_iter = []
    for iteration in range(max_iterations):
        lyapunov_out = nn_lyapunov(e)
        policy_out = nn_policy(e_and_t, zeros_and_t)
        loss = 0
        avg_policy = 0
        avg_u = 0
        for i in range(n):
            e_i = e[i]
            # grd = torch.zeros(u_shape)
            # u_lyapunov_out.backward()
            # e_grad = e.grad
            # derivative_lyapunov_wrt_ei = e_grad[i]
            # e.grad.zero_()
            weights1 = nn_lyapunov.layer1.weight.data
            weights2 = nn_lyapunov.layer2.weight.data
            drv_e1 = torch.sum(weights1[:, 0] * weights2)
            drv_e2 = torch.sum(weights1[:, 1] * weights2)
            derivative_lyapunov_wrt_ei = torch.tensor([drv_e1, drv_e2])
            # print(derivative_lyapunov_wrt_ei)
            f_e_i = f_of_e(e_i[0], e_i[1], t[i], policy_out[i])
            loss = + max(alpha * (abs(e_i[0]) + abs(e_i[1])) - lyapunov_out[i], 0) + \
                   max(torch.linalg.norm(e_i) - b_friction_constant, 0) * \
                   max(torch.inner(derivative_lyapunov_wrt_ei, f_e_i), 0)

        loss = loss / n
        loss_each_iter.append(loss.item())
        print_loss(loss, iteration)
        optimizer_l.zero_grad()
        optimizer_p.zero_grad()
        loss.backward()
        optimizer_l.step()
        optimizer_p.step()

    return nn_lyapunov, nn_policy, loss_each_iter
