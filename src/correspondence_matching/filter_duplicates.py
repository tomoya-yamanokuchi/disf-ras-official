import numpy as np
from value_object import PointNormalUnitPairs, PointNormalIndexUnitPairs
from value_object import CorrespondenceFilteredResult
from value_object import TargetPointNormalIndexPairs


def filter_duplicates(
    source                        : PointNormalIndexUnitPairs,
    target_correspondences        : PointNormalUnitPairs,
    target_correspondences_indices: np.ndarray,
    distances                     : np.ndarray,  # 追加
) -> CorrespondenceFilteredResult:
    """
    同じ object index に対応する correspondence が複数ある場合、
    距離が最も小さい correspondence を優先的に残す。
    """

    unique_indices = np.unique(target_correspondences_indices)

    valid_mask = np.zeros_like(target_correspondences_indices, dtype=bool)

    for idx in unique_indices:
        dup_mask = (target_correspondences_indices == idx)
        dup_positions = np.where(dup_mask)[0]

        # その object 点に対応する correspondence 群の中で距離が最小のものを選択
        best_local = dup_positions[np.argmin(distances[dup_mask])]
        valid_mask[best_local] = True

    # ---- target 側フィルタ ----
    filtered_target = TargetPointNormalIndexPairs(
        points  = target_correspondences.points [valid_mask],
        normals = target_correspondences.normals[valid_mask],
        indices = target_correspondences_indices[valid_mask],
    )

    # ---- source 側フィルタ ----
    filtered_source = PointNormalIndexUnitPairs(
        points         = source.points        [valid_mask],
        normals        = source.normals       [valid_mask],
        finger_indices = source.finger_indices[valid_mask],
    )

    # import ipdb; ipdb.set_trace()

    return CorrespondenceFilteredResult(
        filtered_target = filtered_target,
        filtered_source = filtered_source,
    )
