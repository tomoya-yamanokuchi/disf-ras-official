import numpy as np


def transform_fingertip_normal(
        n_p : np.ndarray, # (num, 3)
        R   : np.ndarray, # (3, 3)
    ):
    # import ipdb ; ipdb.set_trace()
    return np.dot(n_p, R.T)