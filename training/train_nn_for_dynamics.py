import numpy as np


def train(nn, x, t, e, dynamics, f_of_e, d_friction_constant, alpha, max_iterations, optimizer):
    # todo calculate derivative_lyapunov_wrt_input
    n = len(x)
    derivative_lyapunov_wrt_input = 0
    for i in range(max_iterations):
        u_lyapunov_out, control_out = nn(e, t)
        dynamics_out = dynamics(x[0], x[1], u_lyapunov_out)
        # todo calculate loss
        # todo change to vector form
        loss = (1 / n) * sum(max(alpha * (abs(e[0]) + abs(e[1])) - u_lyapunov_out, 0) +
                             max(np.linalg.norm(e) - d_friction_constant, 0) *
                             max(derivative_lyapunov_wrt_input * f_of_e(e, u_lyapunov_out, t)))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
