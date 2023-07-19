import torch

N = 500
e_dim = 2
e_interval_low = -10
e_interval_high = 10
t_dim = 1
t_interval_low = 0
t_interval_high = 50


def sample_data_points():
    e_samples = torch.Tensor(N, e_dim).uniform_(e_interval_low, e_interval_high)
    t_samples = torch.Tensor(N, t_dim).uniform_(t_interval_low, t_interval_high)
    return e_samples, t_samples

