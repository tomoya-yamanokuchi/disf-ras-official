import numpy as np
from value_object import TargetPointNormalIndexPairs
from value_object import PointNormalUnitPairs, PointNormalIndexUnitPairs
from value_object import CorrespondenceFilteredResult


def filter_by_finger_normals(
    filtered_result : CorrespondenceFilteredResult,
    angle_threshold : float,
) -> CorrespondenceFilteredResult:

    filtered_source = filtered_result.filtered_source
    filtered_target = filtered_result.filtered_target

    # ------------------ finger index == 1 ------------------
    finger_mask    = filtered_source.finger_indices == 1
    finger_normals = filtered_source.normals[finger_mask]

    # 平均法線を計算
    finger_mean_normal = finger_normals.mean(axis=0)
    finger_mean_normal /= np.linalg.norm(finger_mean_normal)  # 正規化

    # targetの各法線との角度を計算
    target_normals = filtered_target.normals[finger_mask] # targets corresponding to finger_indices==1
    dot_products   = target_normals @ finger_mean_normal
    angles         = np.arccos(np.clip(dot_products, -1.0, 1.0))  # 角度（ラジアン）に変換
    valid_mask     = angles >= angle_threshold

    # filter: target
    filtered_target_result_finger1 = TargetPointNormalIndexPairs(
        points  = filtered_target.points [finger_mask][valid_mask],
        normals = filtered_target.normals[finger_mask][valid_mask],
        indices = filtered_target.indices[finger_mask][valid_mask],
    )
    # filter: source
    filtered_source_result_finger1 = PointNormalIndexUnitPairs(
        points         = filtered_source.points        [finger_mask][valid_mask],
        normals        = filtered_source.normals       [finger_mask][valid_mask],
        finger_indices = filtered_source.finger_indices[finger_mask][valid_mask],
    )

    # ------------------ finger index == 2 ------------------
    finger_mask2 = filtered_source.finger_indices == 2
    # filter: source
    source_finger2 = PointNormalIndexUnitPairs(
        points         = filtered_source.points        [finger_mask2],
        normals        = filtered_source.normals       [finger_mask2],
        finger_indices = filtered_source.finger_indices[finger_mask2],
    )
    # filter: target
    target_finger2 = TargetPointNormalIndexPairs(
        points  = filtered_target.points [finger_mask2],
        normals = filtered_target.normals[finger_mask2],
        indices = filtered_target.indices[finger_mask2],
    )

    # ------------- concatenate finger 1 and 2 -------------
    # filter: source
    filtered_source_result = PointNormalIndexUnitPairs(
        points         = np.concatenate([filtered_source_result_finger1.points,         source_finger2.points]),
        normals        = np.concatenate([filtered_source_result_finger1.normals,        source_finger2.normals]),
        finger_indices = np.concatenate([filtered_source_result_finger1.finger_indices, source_finger2.finger_indices]),
    )
    # filter: target
    filtered_target_result = TargetPointNormalIndexPairs(
        points  = np.concatenate([filtered_target_result_finger1.points,  target_finger2.points]),
        normals = np.concatenate([filtered_target_result_finger1.normals, target_finger2.normals]),
        indices = np.concatenate([filtered_target_result_finger1.indices, target_finger2.indices]),
    )

    # -------
    # print(filtered_source_result.finger_indices)
    # print(filtered_target_result.indices)
    # import ipdb; ipdb.set_trace()
    # -------

    return CorrespondenceFilteredResult(
        filtered_target = filtered_target_result,
        filtered_source = filtered_source_result,
    )
