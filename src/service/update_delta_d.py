import numpy as np


def update_delta_d(
        delta_d   : np.ndarray,
        delta_d_t : np.ndarray,
    ):
    return (delta_d_t + delta_d)