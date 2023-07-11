import torch
import torch.nn.functional as F


def train(nn, x1, x2, t, dynamics, f_of_e, alpha, max_iterations, learning_rate, optimizer):
    # todo calculate derivative_lyapunov_wrt_input
    derivative_lyapunov_wrt_input = 0
    for i in range(max_iterations):
        u_lyapunov, control_out = nn(x1, x2, t)
        dynamics_out = dynamics(x1, x2, u_lyapunov)
