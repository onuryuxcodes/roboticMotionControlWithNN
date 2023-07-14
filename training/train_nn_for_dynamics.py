import numpy as np
from log_messages import print_loss


def train(nn_lyapunov, nn_porlicy, t, e, f_of_e, d_friction_constant, alpha, max_iterations, optimizer_l, optimizer_p):
    n = len(e)
    e.requires_grad = True
    for iteration in range(max_iterations):
        u_lyapunov_out = nn_lyapunov(e, t)
        e_grad = e.grad
        loss = 0
        for i in range(n):
            e_i = e[i]
            derivative_lyapunov_wrt_ei = e_grad[i]
            loss = + max(alpha * (abs(e_i[0]) + abs(e_i[1])) - u_lyapunov_out, 0) + \
                max(np.linalg.norm(e_i) - d_friction_constant, 0) * \
                max(derivative_lyapunov_wrt_ei * f_of_e(e_i, u_lyapunov_out, t), 0)

        loss = loss/n
        print_loss(loss, iteration)
        # Training mechanics, take a step nn_lyapunov
        optimizer_l.zero_grad()
        loss.backward()
        optimizer_l.step()
        # Take a step nn_policy
        optimizer_p.zero_grad()
        loss.backward()
        optimizer_p.step()
