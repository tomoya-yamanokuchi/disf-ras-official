import numpy as np
from .transform_fingertip_points import transform_fingertip_points
from .transform_fingertip_normal import transform_fingertip_normal
from value_object import PointNormalIndexUnitPairs


def transform_fingertip(
        source : PointNormalIndexUnitPairs,
        # ---
        R      : np.ndarray, # (3, 3)
        t      : np.ndarray, # (3,)
        delta_d: float,
        # ---
        v      : np.ndarray, # (3,)
    ):
    p       = source.points
    n_p     = source.normals
    j_array = source.finger_indices
    # -------
    return PointNormalIndexUnitPairs(
        points         = transform_fingertip_points(p, R, t, delta_d, j_array, v),
        normals        = transform_fingertip_normal(n_p, R),
        finger_indices = j_array,
    )
