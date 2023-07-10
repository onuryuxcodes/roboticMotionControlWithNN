# Dynamics of the Inverted Pendulum
#  x ̇ = f (x, u)
import math
import numpy as np
from dynamics.constants import *


def f_dynamics_state_space(x1, x2, u):
    return (1 / m * math.pow(length, 2)) * m * g * length * \
           np.sin(x1) + u - b_friction * x2
