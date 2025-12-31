from .calculate_centroid import calculate_centroid
from value_object import PointNormalUnitPairs
import numpy as np



def shift_object_center_to_origin(
        target_point_normal: PointNormalUnitPairs,
    ):
    So                = target_point_normal.points
    So_center         = calculate_centroid(So, keepdims=False)
    t_shift_So_center = (np.zeros(3) - So_center)
    # ----
    return PointNormalUnitPairs(
        points         = (So + t_shift_So_center),
        normals        = target_point_normal.normals,
    ), t_shift_So_center
