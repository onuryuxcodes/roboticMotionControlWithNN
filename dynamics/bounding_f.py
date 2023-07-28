import math
from dynamics.constants import *
import torch


#def p_1(t):
#    return 6 * math.exp(-1 * t) + 0.1

def p_1(t):
    return (2*math.pi - 0.2)*math.exp(-t) + 0.2


def derivative_p_1(t):
    return (-2*math.pi + 0.2)*math.exp(-t)


#def derivative_p_1(t):
#    return -6 * math.exp(-1 * t)

def p_2(t):
    return 8*math.exp(-0.5 * t) + 2


def derivative_p_2(t):
    return -4 * math.exp(-0.5 * t)


#def p_2(t):
#    return 10 * math.exp(-0.1 * t) + 0.5


#def derivative_p_2(t):
#    return -1 * math.exp(-0.1 * t)


def term_exp_e(e, c):
    return math.exp(e) + c


def f_e1_term_1(e1, t):
    return 2 / (p_1(t) * (1 - math.pow(term_exp_e(e1, -1) / term_exp_e(e1, 1), 2)))


def f_e2_term_1(e2, t):
    return 2 / (p_2(t) * (1 - math.pow(term_exp_e(e2, -1) / term_exp_e(e2, 1), 2)))


def f_e1(e1, e2, t):
    return f_e1_term_1(e1, t) * (p_2(t) * (term_exp_e(e2, -1) / term_exp_e(e2, 1)) -
                                 derivative_p_1(t) * (term_exp_e(e1, -1) / term_exp_e(e1, 1)))


def f_e2(e1, e2, t, u):
    return f_e2_term_1(e2, t) * ((1 / m_l_square) * (m_g_l * math.sin(p_1(t)*term_exp_e(e1, -1) / term_exp_e(e1, 1)) +
                                                     u - b_friction * p_2(t) * (term_exp_e(e2, -1) / term_exp_e(e2, 1))) -
                                 derivative_p_2(t) * term_exp_e(e2, -1) / term_exp_e(e2, 1))


def f_of_e(e1, e2, t, u):
    return torch.tensor([[f_e1(e1, e2, t), f_e2(e1, e2, t, u)]])

