import torch

N = 500
e_dim = 2
e_interval_low = -5
e_interval_high = 5
t_dim = 1
t_interval_low = 0
t_interval_high = 50
alpha = 1


def sample_data_points(sample_size=N):
    e_samples = torch.Tensor(sample_size, e_dim).uniform_(e_interval_low, e_interval_high)
    t_samples = torch.Tensor(sample_size, t_dim).uniform_(t_interval_low, t_interval_high)
    e_and_t = torch.cat((e_samples, t_samples), -1)
    zeros = torch.zeros(sample_size, e_dim)
    zeros_and_t = torch.cat((zeros, t_samples), -1)
    return e_samples, t_samples, e_and_t, zeros_and_t


def sample_e_only(size):
    e_samples = torch.Tensor(size, e_dim).uniform_(e_interval_low, e_interval_high)
    return e_samples
