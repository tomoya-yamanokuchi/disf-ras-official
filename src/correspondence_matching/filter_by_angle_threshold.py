import numpy as np
from value_object import TargetPointNormalIndexPairs
from value_object import PointNormalUnitPairs, PointNormalIndexUnitPairs
from value_object import CorrespondenceFilteredResult
from value_object import CorrespondenceFilteredResult


def filter_by_angle_threshold(
    filtered_result: CorrespondenceFilteredResult,
    angle_threshold: float,
) -> CorrespondenceFilteredResult:

    filtered_source = filtered_result.filtered_source
    filtered_target = filtered_result.filtered_target

    # finger_indicesごとにデータを分割
    finger1_mask    = filtered_source.finger_indices == 1
    finger2_mask    = filtered_source.finger_indices == 2

    # グループごとの法線ベクトルを取得
    normals_finger1 = filtered_target.normals[finger1_mask]
    normals_finger2 = filtered_target.normals[finger2_mask]

    # ------------------ finger index == 1 ------------------
    # 平均法線を計算
    finger1_mean_normal = normals_finger1.mean(axis=0)
    finger1_mean_normal /= np.linalg.norm(finger1_mean_normal)  # 正規化

    # ------------------ finger index == 2 ------------------
    # finger2の法線との角度を計算
    dot_products         = normals_finger2 @ finger1_mean_normal
    angles               = np.arccos(np.clip(dot_products, -1.0, 1.0))  # 角度（ラジアン）に変換
    valid_finger2_mask   = angles >= angle_threshold

    # filter: target
    filtered_target_result_finger2 = TargetPointNormalIndexPairs(
        points  = filtered_target.points [finger2_mask][valid_finger2_mask],
        normals = filtered_target.normals[finger2_mask][valid_finger2_mask],
        indices = filtered_target.indices[finger2_mask][valid_finger2_mask],
    )
    # filter: source
    filtered_source_result_finger2 = PointNormalIndexUnitPairs(
        points         = filtered_source.points        [finger2_mask][valid_finger2_mask],
        normals        = filtered_source.normals       [finger2_mask][valid_finger2_mask],
        finger_indices = filtered_source.finger_indices[finger2_mask][valid_finger2_mask],
    )

    # ------------------ finger index == 1 ------------------
    finger_mask1 = filtered_source.finger_indices == 1
    # filter: source
    source_finger1 = PointNormalIndexUnitPairs(
        points         = filtered_source.points        [finger_mask1],
        normals        = filtered_source.normals       [finger_mask1],
        finger_indices = filtered_source.finger_indices[finger_mask1],
    )
    # filter: target
    target_finger1 = TargetPointNormalIndexPairs(
        points  = filtered_target.points [finger_mask1],
        normals = filtered_target.normals[finger_mask1],
        indices = filtered_target.indices[finger_mask1],
    )

    # ------------- concatenate finger 1 and 2 -------------
    # filter: source
    filtered_source_result = PointNormalIndexUnitPairs(
        points         = np.concatenate([source_finger1.points,         filtered_source_result_finger2.points]),
        normals        = np.concatenate([source_finger1.normals,        filtered_source_result_finger2.normals]),
        finger_indices = np.concatenate([source_finger1.finger_indices, filtered_source_result_finger2.finger_indices]),
    )
    # filter: target
    filtered_target_result = TargetPointNormalIndexPairs(
        points  = np.concatenate([target_finger1.points,  filtered_target_result_finger2.points]),
        normals = np.concatenate([target_finger1.normals, filtered_target_result_finger2.normals]),
        indices = np.concatenate([target_finger1.indices, filtered_target_result_finger2.indices]),
    )

    # import ipdb ; ipdb.set_trace()
    return CorrespondenceFilteredResult(
        filtered_target = filtered_target_result,
        filtered_source = filtered_source_result,
    )
