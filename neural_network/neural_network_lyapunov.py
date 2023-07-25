from sampling.sampling_const_and_functions import alpha
import torch


class NeuralNetworkLyapunov(torch.nn.Module):

    def __init__(self, input_dim, neurons_hidden_layer, output_dim):
        super(NeuralNetworkLyapunov, self).__init__()
        self.layer1 = torch.nn.Linear(input_dim, neurons_hidden_layer)
        self.layer2 = torch.nn.Linear(neurons_hidden_layer, output_dim)
        # activation unused for now (i.e. layers pure linear activation; y=x)
        self.tanh_activation = torch.nn.Tanh()

    def forward(self, x):
        row_size_x = x.size(dim=0)
        col_size_x = x.size(dim=1)
        e1_tensor_absolute = torch.abs(x[:, 0]).view(row_size_x, 1)
        e2_tensor_absolute = torch.abs(x[:, 1]).view(row_size_x, 1)
        x = self.layer1(x)
        x = self.layer2(x)
        zero_tensor = torch.zeros(row_size_x, col_size_x)
        x_0 = self.layer1(zero_tensor)
        x_0 = self.layer2(x_0)
        alpha_epsilon_value = torch.mul(e1_tensor_absolute + e2_tensor_absolute, alpha)
        x = x - x_0 + alpha_epsilon_value
        return x
