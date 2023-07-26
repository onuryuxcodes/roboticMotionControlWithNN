import torch

N = 500
e_dim = 2
e_interval_low = -5
e_interval_high = 5
t_dim = 1
t_interval_low = 0
t_interval_high = 50
alpha = 1


def sample_data_points():
    e_samples = torch.Tensor(N, e_dim).uniform_(e_interval_low, e_interval_high)
    t_samples = torch.Tensor(N, t_dim).uniform_(t_interval_low, t_interval_high)
    e_and_t = torch.cat((e_samples, t_samples), -1)
    return e_samples, t_samples, e_and_t

