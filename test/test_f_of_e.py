import unittest
from dynamics.bounding_f import f_of_e
from dynamics.bounding_f import p_1
import torch


class TestFofE(unittest.TestCase):

    def test_case1(self):
        u = -1.5
        epsilon_1 = -2.9
        epsilon_2 = 1.5
        t = 4.5
        value = f_of_e(
            e1=epsilon_1,
            e2=epsilon_2,
            u=u,
            t=t
        ).tolist()[0]
        value[0] = round(value[0], 4)
        value[1] = round(value[1], 4)
        self.assertEqual(value, [65.9741, -6.3280])

