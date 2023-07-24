from dynamics.inverted_pendulum import f_dynamics_state_space
from sampling.sampling_const_and_functions import sample_data_points
from neural_network.neural_network_policy import NeuralNetworkControlPolicy
from neural_network.neural_network_lyapunov import NeuralNetworkLyapunov
from training.train_nn_for_dynamics import train
from dynamics.constants import b_friction
from dynamics.bounding_f import f_of_e
import torch

if __name__ == '__main__':
    # x1 = [0, 1]
    # x2 = [0, 1]
    # u = [-1, 1]
    # print(f_dynamics_state_space(x1, x2, u))
    e, t = sample_data_points()
    learning_rate = 0.0009
    neural_network_for_lyapunov = NeuralNetworkLyapunov(
        input_dim=2,
        neurons_hidden_layer=5,
        output_dim=1)
    neural_network_for_control_policy = NeuralNetworkControlPolicy(
        input_dim=3,
        neurons_hidden_layer=5)
    optimizer1 = torch.optim.Adam(neural_network_for_lyapunov.parameters(), lr=learning_rate)
    optimizer2 = torch.optim.Adam(neural_network_for_control_policy.parameters(), lr=learning_rate)
    e.requires_grad = True
    train(
        nn_lyapunov=neural_network_for_lyapunov,
        nn_policy=neural_network_for_control_policy,
        optimizer_l=optimizer1,
        optimizer_p=optimizer2,
        alpha=1,
        max_iterations=1000,
        b_friction_constant=b_friction,
        e=e,
        t=t,
        f_of_e=f_of_e
    )

