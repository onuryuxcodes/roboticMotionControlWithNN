import numpy as np
from training.log_messages import print_loss
import torch


def train(nn_lyapunov, nn_policy, t, e, f_of_e, b_friction_constant, alpha, max_iterations, optimizer_l, optimizer_p):
    n = len(e)
    for iteration in range(max_iterations):
        u_lyapunov_out = nn_lyapunov(e)
        u_shape = u_lyapunov_out.shape
        loss = 0
        for i in range(n):
            e_i = e[i]
            grd = torch.zeros(u_shape)
            u_lyapunov_out.backward(gradient=grd, retain_graph=True)
            e_grad = e.grad
            derivative_lyapunov_wrt_ei = e_grad[i]
            e.grad.zero_()
            # print(derivative_lyapunov_wrt_ei)
            f_e_i = f_of_e(e_i[0], e_i[1], t[i], u_lyapunov_out[i])
            loss = + max(alpha * (abs(e_i[0]) + abs(e_i[1])) - u_lyapunov_out[i], 0) + \
                max(torch.linalg.norm(e_i) - b_friction_constant, 0) * \
                max(torch.inner(derivative_lyapunov_wrt_ei, f_e_i), 0)

        loss = loss / n
        print_loss(loss, iteration)
        weights = nn_lyapunov.hidden_layer.weight.data.numpy()
        print(weights)

        optimizer_l.zero_grad()
        optimizer_p.zero_grad()
        loss.backward()
        optimizer_l.step()
        optimizer_p.step()




