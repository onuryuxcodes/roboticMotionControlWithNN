from dynamics.inverted_pendulum import f_dynamics_state_space
from sampling.sampling_const_and_functions import sample_data_points
from neural_network.neural_network_policy import NeuralNetworkControlPolicy
from neural_network.neural_network_lyapunov import NeuralNetworkLyapunov
from training.train_nn_for_dynamics import train

if __name__ == '__main__':
    # x1 = [0, 1]
    # x2 = [0, 1]
    # u = [-1, 1]
    # print(f_dynamics_state_space(x1, x2, u))
    e, t = sample_data_points()
    neural_network_for_lyapunov = NeuralNetworkLyapunov(
        input_dim=2,
        neurons_hidden_layer=5,
        output_dim=1)
    neural_network_for_control_policy = NeuralNetworkControlPolicy(
        input_dim=3,
        neurons_hidden_layer=5)

