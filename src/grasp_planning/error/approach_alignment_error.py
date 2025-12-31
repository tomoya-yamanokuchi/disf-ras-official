import numpy as np


def approach_alignment_error(
        n_z   : np.ndarray,
        n_app : np.ndarray,
    ):
    # import ipdb; ipdb.set_trace()
    return (np.dot(n_z, n_app) - 1)
