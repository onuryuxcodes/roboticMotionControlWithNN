from training.log_messages import print_loss, print_invalid_points_count
import torch
import numpy as np
from dreal import *
from sampling.sampling_const_and_functions import \
    e_interval_high, \
    e_interval_low, \
    t_interval_high, \
    t_interval_low


def check_satisfies_all_domain_with_dreal(nn_lyapunov, nn_policy, d, f_of_e, gamma, derivative_lyapunov_wrt_ei):
    config = Config()
    config.use_polytope_in_forall = True
    config.use_local_optimization = True
    config.precision = 1e-2

    e1 = Variable("e1")
    e2 = Variable("e2")
    t = Variable("t")

    bound_sat = And(e_interval_low <= e1, e1 <= e_interval_high,
                e_interval_low <= e2, e2 <= e_interval_high,
                t <= t_interval_low, t <= t_interval_high)

    cond1 = Not(e1 ** 2 + e2 ** 2 > gamma, nn_lyapunov(torch.tensor([[e1, e2]])).item() <= 0)
    CheckSatisfiability(And(bound_sat, cond1), config)


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
            check_satisfies_all_domain_with_dreal(nn_policy, d, f_of_e, gamma, derivative_lyapunov_wrt_ei)

    return nn_lyapunov, nn_policy, loss_each_iter
