import torch


class NeuralNetworkLyapunov(torch.nn.Module):

    def __init__(self, input_dim, neurons_hidden_layer, output_dim):
        super(NeuralNetworkLyapunov, self).__init__()
        self.layer1 = torch.nn.Linear(input_dim, 1)
        # self.hidden_layer = torch.nn.Linear(neurons_hidden_layer, output_dim)
        self.tanh_activation = torch.nn.Tanh()

    def forward(self, x):
        x = self.layer1(x)
        # lyapunov_z = self.tanh_activation(x)
        return x
