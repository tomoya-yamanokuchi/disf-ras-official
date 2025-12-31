import numpy as np




def generate_surface_panda_hand(
        num_fingertip_surface_points: int,
        d0                          : float,
        finger_index                : int,
    ):
    # assert j in [1, 2]
    # -------------------
    # mujoco_gripper_init_qpos =
    # y_finger_offset          = ((-1)**j) * mujoco_gripper_init_qpos
    y_finger_offset = ((-1)**finger_index) * (0.5 * d0)
    # -------------------
    num_surface_points_z = int(np.sqrt(num_fingertip_surface_points)) # hyper-parameter
    num_surface_points_y = int(np.sqrt(num_fingertip_surface_points)) # hyper-parameter
    # -------------------
    # import ipdb ; ipdb.set_trace()
    points = []
    for i in range(num_surface_points_z):
        for k in range(num_surface_points_y):
            point = np.array([0.003*(k-2), y_finger_offset, 0.003 * (i-2)])  # 仮想オフセットを適用
            points.append(point)
    # -------------------
    return np.array(points)


if __name__ == '__main__':

