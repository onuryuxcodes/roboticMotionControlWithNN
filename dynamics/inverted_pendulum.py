# Dynamics of the Inverted Pendulum
#  x ̇ = f (x, u)
import math
import numpy as np
from dynamics.constants import *


def f_dynamics_state_space(x1, x2, u):
    """
    x1: θ, the pendulum’s angle from the downward position
    x2: θ ̇, derivative of θ, the pendulum’s angular velocity
    u: control input
    """
    return (1 / m * math.pow(length, 2)) * m * g * length * \
           rnp.sin(x1) + u - b_friction * x2
