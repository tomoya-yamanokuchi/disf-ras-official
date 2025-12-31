import numpy as np
from value_object import PointNormalUnitPairs

def point2point_error(
        source  : PointNormalUnitPairs,
        target  : PointNormalUnitPairs,
    ):
    source_points  = source.points
    target_points  = target.points
    # ---
    return np.sqrt(np.sum((source_points - target_points)**2, axis=-1))
