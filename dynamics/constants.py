g = 9.81  # gravity
length = 1  # length of pole
m = 1  # ball mass
b_friction = 1  # friction
m_l_square = m*length*length
m_g_l = m*g*length

col_loss_each_iter = "loss"
col_u_each_iter = "avg_u"
col_policy_each_iter = "avg_policy"

column_list = [
        col_loss_each_iter,
        col_policy_each_iter,
        col_u_each_iter
    ]
