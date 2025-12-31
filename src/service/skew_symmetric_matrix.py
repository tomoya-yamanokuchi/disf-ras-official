import numpy as np


def skew_symmetric_matrix(r: np.ndarray):
    """
        r : axis-angle vector (3,)
    """
    assert r.shape == (3,)
    # ---
    return np.array([
        [  0  , -r[2],  r[1]],
        [ r[2],    0 , -r[0]],
        [-r[1],  r[0],    0 ],
    ])