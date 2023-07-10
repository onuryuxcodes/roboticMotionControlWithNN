from dynamics.inverted_pendulum import f_dynamics_state_space

if __name__ == '__main__':
    x1 = [0, 1]
    x2 = [0, 1]
    u = [-1, 1]
    print(f_dynamics_state_space(x1, x2, u))
