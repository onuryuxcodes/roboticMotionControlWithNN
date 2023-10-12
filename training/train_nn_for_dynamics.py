from training.log_messages import print_loss, print_invalid_points_count, print_total_number_of_points
import torch
import numpy as np
from sampling.sampling_const_and_functions import sample_data_points


# Falsification procedure
def check_lyapunov_validity(nn_lyapunov, nn_policy, t, e, d, e_and_t,
                            zeros_and_t, f_of_e, gamma, derivative_lyapunov_wrt_ei,
                            index_list_counter_example=[],
                            verbose=False):
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
    if verbose:
        print_invalid_points_count(n, invalid_points_count)
    return new_counter_examples == 0, torch.as_tensor(new_counter_examples)


def calculate_derivative_wrt_ei(nn_lyapunov):
    weights1 = nn_lyapunov.layer1.weight.data
    weights2 = nn_lyapunov.layer2.weight.data
    drv_e1 = torch.sum(weights1[:, 0] * weights2)
    drv_e2 = torch.sum(weights1[:, 1] * weights2)
    return torch.tensor([drv_e1, drv_e2])


def train(nn_lyapunov, nn_policy, t, e, e_and_t, zeros_and_t, f_of_e, b_friction_constant, d,
          alpha, max_iterations,
          optimizer_l,
          optimizer_p):
    loss_each_iter = []
    training_data_each_iter = []
    index_list_counter_example = []
    number_of_invalid_points_each_iter = []
    is_valid = False
    training_iteration = 0
    gamma = 0.0001
    while training_iteration < max_iterations and not is_valid:
        lyapunov_out = nn_lyapunov(e)
        policy_out = nn_policy(e_and_t, zeros_and_t)
        derivative_lyapunov_wrt_ei = calculate_derivative_wrt_ei(nn_lyapunov)

        test_is_valid, invalid_data_point_index = \
            check_lyapunov_validity(nn_lyapunov, nn_policy, t, e, d, e_and_t,
                                    zeros_and_t, f_of_e, gamma, derivative_lyapunov_wrt_ei,
                                    index_list_counter_example)
        number_of_invalid_points_each_iter.append(len(invalid_data_point_index))
        n = len(e)
        training_data_each_iter.append(n)
        loss_i = 0
        for i in range(n):
            e_i = e[i]
            f_e_i = f_of_e(e_i[0], e_i[1], t[i], policy_out[i])
            e_i_norm_minus_d = torch.subtract(torch.linalg.norm(e_i), d)
            mult_term = torch.mul(torch.max(e_i_norm_minus_d, torch.zeros_like(e_i[0])), torch.max(
                torch.inner(derivative_lyapunov_wrt_ei, f_e_i), torch.zeros_like(e_i[0])))
            max_term1 = torch.subtract(torch.mul(alpha, torch.add(torch.abs(e_i[0]), torch.abs(e_i[1]))),
                                       lyapunov_out[i])
            loss_i = torch.add(torch.add(torch.max(max_term1, torch.zeros_like(e_i[0])), mult_term), loss_i)
        loss = torch.div(loss_i, n)
        optimizer_l.zero_grad()
        optimizer_p.zero_grad()
        loss.backward()
        optimizer_l.step()
        optimizer_p.step()
        loss_each_iter.append(loss.item())
        training_iteration += 1
        if training_iteration % 100 == 0:
            print_loss(loss, training_iteration)
            print_total_number_of_points(n)
            # Sample new t and e
            test_e, test_t, test_concatenated_e_t, test_zeros_and_t = sample_data_points(sample_size=10)
            test_is_valid, invalid_data_point_index = \
                check_lyapunov_validity(nn_lyapunov, nn_policy, test_t, test_e, d, test_concatenated_e_t,
                                        test_zeros_and_t, f_of_e, gamma, derivative_lyapunov_wrt_ei,
                                        index_list_counter_example,
                                        verbose=True)
            if not test_is_valid:
                # Points to be added to the training samples
                new_e = torch.index_select(test_e, 0, invalid_data_point_index)
                new_t = torch.index_select(test_t, 0, invalid_data_point_index)
                new_concatenated_e_t = torch.index_select(test_concatenated_e_t, 0, invalid_data_point_index)
                new_zeros_and_t = torch.index_select(test_zeros_and_t, 0, invalid_data_point_index)
                # Concat points to original training data tensor
                e = torch.cat((e, new_e), 0)
                t = torch.cat((t, new_t), 0)
                e_and_t = torch.cat((e_and_t, new_concatenated_e_t), 0)
                zeros_and_t = torch.cat((zeros_and_t, new_zeros_and_t), 0)
                is_valid = False
            else:
                is_valid, _ = check_lyapunov_validity(nn_lyapunov, nn_policy, t, e, d, e_and_t,
                                                      zeros_and_t, f_of_e, gamma, derivative_lyapunov_wrt_ei,
                                                      index_list_counter_example)
    return nn_lyapunov, nn_policy, loss_each_iter, training_data_each_iter, number_of_invalid_points_each_iter
