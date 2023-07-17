import torch


class NeuralNetworkLyapunov(torch.nn.Module):

    def __init__(self, input_dim, neurons_hidden_layer, output_dim):
        super(NeuralNetworkLyapunov, self).__init__()
        self.input_layer = torch.nn.Linear(input_dim, neurons_hidden_layer)
        self.hidden_layer = torch.nn.Linear(neurons_hidden_layer, output_dim)
        self.tanh_activation = torch.nn.Tanh()

    def forward(self, x):
        x = self.input_layer(x)
        x = self.tanh_activation(x)
        x = self.hidden_layer(x)
        lyapunov_z = self.tanh_activation(x)
        return lyapunov_z
