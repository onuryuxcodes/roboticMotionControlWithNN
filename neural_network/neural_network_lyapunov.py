import torch


class FeedForwardNeuralNetworkOneHiddenLayer(torch.nn.Module):

    def __init__(self, input_dim, neurons_hidden_layer, output_dim):
        super(FeedForwardNeuralNetworkOneHiddenLayer, self).__init__()
        # torch.manual_seed(2)
        self.input_layer = torch.nn.Linear(input_dim, neurons_hidden_layer)
        self.hidden_layer = torch.nn.Linear(neurons_hidden_layer, output_dim)
        self.independent_control_layer = torch.nn.Linear(input_dim, 1, bias=False)
        self.tanh_activation = torch.nn.Tanh()
        # self.control.weight = torch.nn.Parameter(lqr)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.tanh_activation(x)
        x = self.hidden_layer(x)
        lyapunov_z = self.tanh_activation(x)
        u = self.independent_control_layer(x)
        return lyapunov_z, u

