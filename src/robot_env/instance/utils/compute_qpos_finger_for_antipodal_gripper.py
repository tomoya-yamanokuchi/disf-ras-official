import numpy as np


def compute_qpos_finger_for_antipodal_gripper(
        d0     : float,
        d_min  : float,
        delta_d: float,
        d_bias : float,
    ):
    d_opt = (d0 + delta_d)
    qpos  = 0.5 * (d_opt - (d_min + d_bias)) # 開閉幅=0の時でもfingertipパッドのデフォルトの厚み分だけのオフセットが存在するので考慮する必要あり
    # import ipdb; ipdb.set_trace()
    return qpos
