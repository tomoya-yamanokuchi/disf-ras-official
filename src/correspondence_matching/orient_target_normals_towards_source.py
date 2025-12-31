import numpy as np
from value_object import TargetPointNormalIndexPairs
from value_object import CorrespondenceFilteredResult


def orient_target_normals_towards_source(filtered_result: CorrespondenceFilteredResult) -> CorrespondenceFilteredResult:
    """
    距離ベースで得た対応点ペア (source, target) に対して、
    target 側の法線の符号だけを、source 側の法線と同じ向きになるようにそろえる。

    ここでは「方向」は変えず、「±」の符号だけをいじる。
    """
    src = filtered_result.filtered_source
    tgt = filtered_result.filtered_target

    src_normals = src.normals          # shape (N, 3)
    tgt_normals = tgt.normals.copy()   # shape (N, 3)   ← コピーするのが安全

    # 各対応ペアごとの内積 < 0 のものを反転
    dots = np.einsum("ij,ij->i", src_normals, tgt_normals)
    flip_mask = dots > 0.0

    tgt_normals[flip_mask] *= -1.0

    # NamedTuple / dataclass どちらかで変わりますが、だいたいこんな感じ
    tgt_oriented = TargetPointNormalIndexPairs(
        points  = tgt.points,
        normals = tgt_normals,
        indices = tgt.indices,
    )

    return CorrespondenceFilteredResult(
        filtered_source                 = src,
        filtered_target                 = tgt_oriented,
    )
