from training.log_messages import print_loss, print_invalid_points_count
import torch
import numpy as np


# Falsification procedure
def check_lyapunov_validity(nn_lyapunov, nn_policy, t, e, d, e_and_t,
                            zeros_and_t, f_of_e, gamma, derivative_lyapunov_wrt_ei,
                            index_list_counter_example=[]):
    n = len(e)
    invalid_points_count = 0
    new_counter_examples = []
    if len(index_list_counter_example) == 0:
        index_list_counter_example = np.arange(n).tolist()
    for i in index_list_counter_example:
        e_i1 = e[i][0]
        e_i2 = e[i][1]
        policy_out = nn_policy(e_and_t, zeros_and_t)
        f_e_i = f_of_e(e_i1, e_i2, t[i], policy_out[i])
        is_falsified_cond1 = (e_i1 ** 2 + e_i2 ** 2).item() > gamma and nn_lyapunov(e)[i].item() <= 0
        is_falsified_cond2 = (e_i1 ** 2 + e_i2 ** 2).item() > d and torch.inner(derivative_lyapunov_wrt_ei,
                                                                                f_e_i).item() >= 0
        if is_falsified_cond1 or is_falsified_cond2:
            invalid_points_count += 1
            new_counter_examples.append(i)
    print_invalid_points_count(n, invalid_points_count)
    return new_counter_examples == 0, new_counter_examples


def calculate_derivative_wrt_ei(nn_lyapunov):
    weights1 = nn_lyapunov.layer1.weight.data
    weights2 = nn_lyapunov.layer2.weight.data
    drv_e1 = torch.sum(weights1[:, 0] * weights2)
    drv_e2 = torch.sum(weights1[:, 1] * weights2)
    # print(" drv > " + str(drv_e1.item()) + " " + str(drv_e2.item()))
    return torch.tensor([drv_e1, drv_e2])


def train(nn_lyapunov, nn_policy, t, e, e_and_t, zeros_and_t, f_of_e, b_friction_constant, d,
          alpha, max_iterations,
          optimizer_l,
          optimizer_p):
    n = len(e)
    loss_each_iter = []
    index_list_counter_example = []
    is_valid = False
    training_iteration = 0
    gamma = 0.0001
    while training_iteration < max_iterations and not is_valid:
        lyapunov_out = nn_lyapunov(e)
        policy_out = nn_policy(e_and_t, zeros_and_t)
        loss = 0
        derivative_lyapunov_wrt_ei = calculate_derivative_wrt_ei(nn_lyapunov)

        for i in range(n):
            e_i = e[i]
            f_e_i = f_of_e(e_i[0], e_i[1], t[i], policy_out[i])
            loss += torch.max(alpha * (torch.abs(e_i[0]) + torch.abs(e_i[1])) - lyapunov_out[i],
                              torch.zeros_like(e_i[0])) + \
                    torch.max(torch.linalg.norm(e_i) - d, torch.zeros_like(e_i[0])) * \
                    torch.max(torch.inner(derivative_lyapunov_wrt_ei, f_e_i), torch.zeros_like(e_i[0]))

        loss = loss / n
        loss_each_iter.append(loss.item())
        optimizer_l.zero_grad()
        optimizer_p.zero_grad()
        loss.backward()
        optimizer_l.step()
        optimizer_p.step()
        training_iteration += 1
        if training_iteration % 100 == 0:
            print_loss(loss, training_iteration)
            is_falsified, index_list_counter_example = \
                check_lyapunov_validity(nn_lyapunov, nn_policy, t, e, d, e_and_t,
                                        zeros_and_t, f_of_e, gamma, derivative_lyapunov_wrt_ei,
                                        index_list_counter_example)

    return nn_lyapunov, nn_policy, loss_each_iter
