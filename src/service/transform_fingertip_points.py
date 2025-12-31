import numpy as np


def transform_fingertip_points(
        p      : np.ndarray, # (num, 3)
        # ---
        R      : np.ndarray, # (3, 3)
        t      : np.ndarray, # (3,)
        delta_d: float,
        # ---
        j_array: np.ndarray, # (num, 3)
        v      : np.ndarray, # (3,)
    ):
    # --------------------------------------
    Rp    = np.dot(p, R.T)                             # (50, 3)
    alpha = 0.5 * delta_d * ((-1)**j_array)[:, None]   # (50, 1)
    Rv    = (R @ v)[None, :]                           #  (1, 3)
    # --------------------------------------
    # import ipdb ; ipdb.set_trace()
    return (Rp + t[None, :] + (alpha * Rv))
