import numpy as np


def transform_gripper(
        source_points     : np.ndarray, # (num, 3)
        finger_indices    : np.ndarray, # (num, 3)
        gripper_normal    : np.ndarray, # (3,)
        # ---
        rotation_matrix   : np.ndarray, # (3, 3)
        translation       : np.ndarray, # (3,)
        delta_d           : float,
    ):
    # --------------------------------------
    v     = gripper_normal
    Rp    = np.dot(source_points, rotation_matrix.T)                 # (50, 3)
    t     = translation.reshape(1, -1)                               #  (1, 3)
    alpha = 0.5 * delta_d * ((-1)**finger_indices).reshape(-1, 1)    # (50, 1)
    Rv    = (rotation_matrix @ v).reshape(1, -1)                     #  (1, 3)
    # --------------------------------------
    # import ipdb ; ipdb.set_trace()
    return (Rp + t + (alpha * Rv))