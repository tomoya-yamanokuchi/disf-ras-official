import numpy as np
from value_object import PointNormalUnitPairs

def normal_alignment_error(
        source  : PointNormalUnitPairs,
        target  : PointNormalUnitPairs,
    ):
    source_normals = source.normals
    target_normals = target.normals
    # ---
    assert (len(source_normals.shape) == 2) and (source_normals.shape[1] == 3)
    assert (len(target_normals.shape) == 2) and (target_normals.shape[1] == 3)
    # ---
    # 各点の法線に投影されたエラーを計算
    # import ipdb ; ipdb.set_trace()
    return (np.sum(source_normals * target_normals, axis=1) + 1) # shape = (num,)
