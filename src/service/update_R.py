import numpy as np


def update_R(
        R : np.ndarray,
        Rt: np.ndarray,
    ):
    return (R @ Rt)

