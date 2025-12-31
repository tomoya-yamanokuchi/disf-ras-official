import numpy as np

def compute_pregrasp_from_grasp(
        R_WG_des: np.ndarray,
        t_WG_des: np.ndarray,
        radius  : float
    ):
    """
    R_WG_des : (3,3) world ← gripper の回転行列（MuJoCoのxmat相当）
    t_WG_des : (3,)  最適 grasp の位置（world）
    radius   : pre-grasp まで離す距離 [m]

    グリッパローカル -z 方向から物体に近づくと仮定し、
    その -z 軸方向に radius だけ離れた位置を pre-grasp として返す。
    """

    # ローカル +z軸が world で向いている方向
    z_world = R_WG_des[:, 2]           # shape (3,)

    # ローカル -z 方向（接近方向）
    approach_dir_world = -z_world      # もう正規化済みだが念のため
    # approach_dir_world /= np.linalg.norm(approach_dir_world)

    # pre-grasp の位置
    t_pre = t_WG_des + radius * approach_dir_world

    # 姿勢は grasp と同じ向き
    R_pre = R_WG_des.copy()

    # import ipdb; ipdb.set_trace()

    return R_pre, t_pre
