import numpy as np
from .skew_symmetric_matrix import skew_symmetric_matrix


def small_angle_rotation_matix(r: np.ndarray):
    return np.eye(3) + skew_symmetric_matrix(r)