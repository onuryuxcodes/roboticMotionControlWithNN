import torch


class NeuralNetworkControlPolicy(torch.nn.Module):

    def __init__(self, input_dim, neurons_hidden_layer):
        super(NeuralNetworkControlPolicy, self).__init__()
        self.input_layer = torch.nn.Linear(input_dim, neurons_hidden_layer)
        self.control_layer = torch.nn.Linear(input_dim, 1, bias=False)

    def forward(self, x):
        return self.control_layer(x)

