import numpy as np
import math

def _skew(v):
    """3D ベクトル v から skew-symmetric 行列 [v]_x を作る"""
    x, y, z = v
    return np.array([[ 0.0, -z,   y ],
                     [  z,  0.0, -x ],
                     [ -y,   x,  0.0]], dtype=float)


def axis_angle_to_matrix(axis, angle):
    """
    axis-angle -> 回転行列 (Rodrigues の公式)

    Parameters
    ----------
    axis : array_like, shape (3,)
        回転軸ベクトル（ゼロ以外なら OK。内部で正規化する）
    angle : float
        回転角 [rad]

    Returns
    -------
    R : ndarray, shape (3, 3)
        回転行列
    """
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        raise ValueError("Axis vector must be non-zero.")
    u = axis / n  # 単位ベクトルへ正規化

    K = _skew(u)
    I = np.eye(3)
    s = math.sin(angle)
    c = math.cos(angle)

    # Rodrigues の回転公式
    R = I + s * K + (1.0 - c) * (K @ K)
    return R


def matrix_to_axis_angle(R):
    """
    回転行列 -> axis-angle

    Parameters
    ----------
    R : array_like, shape (3, 3)
        回転行列

    Returns
    -------
    axis : ndarray, shape (3,)
        単位長の回転軸ベクトル
    angle : float
        回転角 [rad]
    """
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        raise ValueError("R must be 3x3.")

    # 回転角 theta = arccos((trace(R) - 1) / 2)
    tr = np.trace(R)
    cos_theta = (tr - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = math.acos(cos_theta)

    # ほぼ恒等回転
    if theta < 1e-8:
        return np.array([1.0, 0.0, 0.0]), 0.0

    # sin(theta) ≈ 0 のとき（theta ≈ pi 付近）は固有ベクトルで軸を取る
    if abs(math.sin(theta)) < 1e-6:
        eigvals, eigvecs = np.linalg.eig(R)
        idx = np.argmin(np.abs(eigvals - 1.0))
        axis = np.real(eigvecs[:, idx])
        axis /= np.linalg.norm(axis)
        return axis, theta

    # 一般の場合
    ux = (R[2, 1] - R[1, 2]) / (2.0 * math.sin(theta))
    uy = (R[0, 2] - R[2, 0]) / (2.0 * math.sin(theta))
    uz = (R[1, 0] - R[0, 1]) / (2.0 * math.sin(theta))
    axis = np.array([ux, uy, uz], dtype=float)
    axis /= np.linalg.norm(axis)
    return axis, theta


def compose_axis_angles(axes, angles):
    """
    複数の axis-angle をまとめて 1 つの axis-angle に変換する。

    回転の適用順は：
        v' = R_1 v
        v'' = R_2 v'
        ...
    という「axes[i], angles[i] の回転を i=0,1,2,... の順にかける」ものとする。

    Parameters
    ----------
    axes : sequence of array_like, each shape (3,)
        各回転の軸ベクトル
    angles : sequence of float
        各回転の角度 [rad]

    Returns
    -------
    axis : ndarray, shape (3,)
        合成した回転の軸
    angle : float
        合成した回転の角度 [rad]
    """
    if len(axes) != len(angles):
        raise ValueError("axes and angles must have the same length.")

    # 合成回転行列 R = R_n ... R_2 R_1
    R = np.eye(3)
    for axis, angle in zip(axes, angles):
        R_step = axis_angle_to_matrix(axis, angle)
        R = R_step @ R  # 右側が先に適用される列ベクトル v 前提

    return matrix_to_axis_angle(R)
