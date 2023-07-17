import math
from constants import *


def p_1(t):
    return 6 * math.exp(-1 * t) + 0.1


def derivative_p_1(t):
    return -6 * math.exp(-1 * t)


def p_2(t):
    return 10 * math.exp(-0.1 * t) + 0.5


def derivative_p_2(t):
    return -1 * math.exp(-0.1 * t)


def term_exp_e(e, c):
    return math.exp(e) + c


def f_e1_term_1(e1, t):
    return 2 / (p_1(t) * (1 - math.pow(term_exp_e(e1, -1) / term_exp_e(e1, 1), 2)))


def f_e2_term_1(e2, t):
    return 2 / (p_2(t) * (1 - math.pow(term_exp_e(e2, -1) / term_exp_e(e2, 1), 2)))


def f_e1(e1, e2, t):
    return f_e1_term_1(e1, t) * (p_2(t) * (term_exp_e(e2, -1) / term_exp_e(e2, 1)) -
                                 derivative_p_1(t) * (term_exp_e(e1, -1) / term_exp_e(e1, 1)))


def f_e2(e2, t, u):
    return f_e2_term_1(e2, t) * ((1 / m_l_square) * (m_g_l * math.sin(term_exp_e(e2, -1) / term_exp_e(e2, 1)) +
                                                     u - b_friction * (term_exp_e(e2, -1) / term_exp_e(e2, 1))) -
                                 derivative_p_2(t) * term_exp_e(e2, -1) / term_exp_e(e2, 1))


def f_of_e(e1, e2, t, u):
    return f_e1(e1, e2, t), f_e2(e2, t, u)
