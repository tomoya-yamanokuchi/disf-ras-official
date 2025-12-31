import numpy as np
from value_object import PointNormalUnitPairs
from .rigid_transformation_from_Rt import rigid_transformation_from_Rt


def point_normal_rigid_transformation_from_Rt(
        point_normal: PointNormalUnitPairs,
        R           : np.ndarray, # (3, 3)
        t           : np.ndarray, # (3,)
    ):
    # ----
    p     = point_normal.points
    n_p   = point_normal.normals
    # ----
    p_p   = rigid_transformation_from_Rt(p, R, t)
    n_p_p = rigid_transformation_from_Rt(n_p, R, t=np.zeros(3))
    # ----
    return PointNormalUnitPairs(
        points  = p_p,
        normals = n_p_p,
    )
