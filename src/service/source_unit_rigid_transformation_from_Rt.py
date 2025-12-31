import numpy as np
from value_object import PointNormalUnitPairs, PointNormalIndexUnitPairs
from .point_normal_rigid_transformation_from_Rt import point_normal_rigid_transformation_from_Rt


def source_unit_rigid_transformation_from_Rt(
        source_unit : PointNormalIndexUnitPairs,
        R           : np.ndarray, # (3, 3)
        t           : np.ndarray, # (3,)
    ):
    # ----
    source_unit_trans = point_normal_rigid_transformation_from_Rt(
        point_normal = PointNormalUnitPairs(
            points  = source_unit.points,
            normals = source_unit.normals,
        ),
        R = R,
        t = t,
    )
    # ----
    return PointNormalIndexUnitPairs(
        points         = source_unit_trans.points,
        normals        = source_unit_trans.normals,
        finger_indices = source_unit.finger_indices,
    )
