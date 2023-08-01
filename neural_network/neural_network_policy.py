import torch


class NeuralNetworkControlPolicy(torch.nn.Module):

    def __init__(self, input_dim, neurons_hidden_layer):
        super(NeuralNetworkControlPolicy, self).__init__()
        self.input_layer = torch.nn.Linear(input_dim, neurons_hidden_layer)
        self.control_layer = torch.nn.Linear(neurons_hidden_layer, 1, bias=False)

    def forward(self, x, zero_tensor):
        x = self.input_layer(x)
        x = self.control_layer(x)
        x_0 = self.input_layer(zero_tensor)
        x_0 = self.control_layer(x_0)
        return x-x_0

