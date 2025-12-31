import numpy as np


def rigid_transformation_from_Rt(
        p: np.ndarray, # (num, 3)
        R: np.ndarray, # (3, 3)
        t: np.ndarray, # (3,)
    ):
    return (np.dot(p, R.T)   + t[None, :])
