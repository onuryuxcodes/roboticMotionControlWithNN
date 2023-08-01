from training.log_messages import print_loss
import torch


# Falsification procedure
def check_lyapunov_validity(nn_lyapunov, nn_policy, t, e, d, e_and_t,
                            zeros_and_t, f_of_e, gamma, derivative_lyapunov_wrt_ei):
    n = len(e)
    for i in range(n):
        e_i = e[i]
        e_i1 = e[i][0]
        e_i2 = e[i][1]
        policy_out = nn_policy(e_and_t, zeros_and_t)
        f_e_i = f_of_e(e_i1, e_i2, t[i], policy_out[i])
        is_valid_cond1 = (e_i1 ** 2 + e_i2 ** 2 > gamma) and nn_lyapunov(e_i) <= 0
        is_valid_cond2 = (e_i1 ** 2 + e_i2 ** 2 > d) and (torch.inner(derivative_lyapunov_wrt_ei, f_e_i))
        if not (is_valid_cond1 and is_valid_cond2):
            return False
    return True


def train(nn_lyapunov, nn_policy, t, e, e_and_t, zeros_and_t, f_of_e, b_friction_constant, d,
          alpha, max_iterations,
          optimizer_l,
          optimizer_p):
    n = len(e)
    loss_each_iter = []
    for iteration in range(max_iterations):
        lyapunov_out = nn_lyapunov(e)
        policy_out = nn_policy(e_and_t, zeros_and_t)
        loss = 0
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
                   max(torch.linalg.norm(e_i) - d, 0) * \
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
