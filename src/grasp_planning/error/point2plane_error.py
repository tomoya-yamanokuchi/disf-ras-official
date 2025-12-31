import numpy as np
from value_object import PointNormalUnitPairs

def point2plane_error(
        source  : PointNormalUnitPairs,
        target  : PointNormalUnitPairs,
    ):
    source_points  = source.points
    target_points  = target.points
    target_normals = target.normals
    # ---
    assert (len(source_points.shape) == 2) and (source_points.shape[1] == 3)
    assert (len(target_points.shape) == 2) and (target_points.shape[1] == 3)
    assert (len(target_normals.shape) == 2) and (target_normals.shape[1] == 3)
    # ---
    # 各点の法線に投影されたエラーを計算
    return np.sum((source_points - target_points) * target_normals, axis=1) # shape = (num,)
