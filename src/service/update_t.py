import numpy as np


def update_t(
        t   : np.ndarray,
        R   : np.ndarray,
        t_t  : np.ndarray,
    ):
    return (R @ t_t) + t