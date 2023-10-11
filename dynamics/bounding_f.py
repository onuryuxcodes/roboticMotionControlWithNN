import math
from dynamics.constants import *
import torch


# Perform the operations through torch so their derivatives are traceable
# when doing gradient descent
def p_1(t):
    mult_term = 2*math.pi - 0.2
    return torch.add(torch.mul(torch.mul(t, -1), mult_term), 0.2)


def derivative_p_1(t):
    return torch.mul(torch.exp(torch.mul(t, -1)), (-2*math.pi + 0.2))


def p_2(t):
    t_05 = torch.mul(t, -0.5)
    return torch.add(torch.mul(torch.exp(t_05), 8), 2)


def derivative_p_2(t):
    t_05 = torch.mul(t, -0.5)
    return torch.mul(torch.exp(t_05), -4)


def term_exp_e(e, c):
    return torch.add(torch.exp(e), c)


def f_e1_term_1(e1, t):
    term_div = torch.div(term_exp_e(e1, -1), term_exp_e(e1, 1))
    term_div_minus = torch.mul(term_div, -1)
    final_term = torch.pow(torch.mul(p_1(t), torch.add(term_div_minus, 1)), -1)
    return torch.mul(final_term, 2)


def f_e2_term_1(e2, t):
    term_div = torch.div(term_exp_e(e2, -1), term_exp_e(e2, 1))
    term_div_minus = torch.mul(term_div, -1)
    final_term = torch.pow(torch.mul(p_2(t), torch.add(term_div_minus, 1)), -1)
    return torch.mul(final_term, 2)


def f_e1(e1, e2, t):
    div_term = torch.div(term_exp_e(e2, -1), term_exp_e(e2, 1))
    left_term = torch.mul(f_e1_term_1(e1, t), torch.mul(p_2(t), div_term))
    div_term2 = torch.div(term_exp_e(e1, -1), term_exp_e(e1, 1))
    right_term = torch.mul(derivative_p_1(t), div_term2)
    return torch.subtract(left_term, right_term)


def f_e2(e1, e2, t, u):
    m_l_square_inverse = 1 / m_l_square
    sin_term = torch.sin(torch.div(torch.mul(p_1(t), term_exp_e(e1, -1)), term_exp_e(e1, 1)))
    sin_term_mult_mgl = torch.mul(sin_term, m_g_l)
    div_term = torch.div(term_exp_e(e2, -1), term_exp_e(e2, 1))
    div_term_mult = torch.mul(p_2(t), div_term)
    div_term_mult_b = torch.mul(torch.mul(div_term_mult, b_friction), -1)
    inner_term = torch.mul(torch.add(torch.add(sin_term_mult_mgl, u), div_term_mult_b), m_l_square_inverse)
    div_term2 = torch.div(term_exp_e(e2, -1), term_exp_e(e2, 1))
    div_term2_mult = torch.mul(derivative_p_2(t), div_term2)
    sub_term = torch.subtract(inner_term, div_term2_mult)
    return torch.mul(f_e2_term_1(e2, t), sub_term)


def f_of_e(e1, e2, t, u):
    # concat vertically
    return torch.cat((f_e1(e1, e2, t), f_e2(e1, e2, t, u)), 0)

