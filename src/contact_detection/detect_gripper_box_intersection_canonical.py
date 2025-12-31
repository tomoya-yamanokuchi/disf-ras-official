# contact_detection.py
import numpy as np

def detect_gripper_box_intersection_canonical(
        point_cloud_G: np.ndarray,
        fingertip_width: float,
        fingertip_height: float,
        gripper_aperture: float,
        margin: float = 0.0,
) -> np.ndarray:
    """
    canonical gripper surface frame G 上で、
    サイズ (fingertip_width × fingertip_height × gripper_aperture)
    の軸揃え box 内に入る点を抽出する。

    G の前提:
      - 原点: 左右 fingertip の中央
      - y軸: finger の開閉方向
      - x軸: fingertip の横幅方向
      - z軸: fingertip の高さ方向

    パラメータ:
      point_cloud_G   : (N,3) np.ndarray, G フレームでの点群
      fingertip_width : float,  x方向の全幅
      fingertip_height: float,  z方向の全高
      gripper_aperture: float,  finger 内面間距離 (y方向の長さ)
      margin          : float,  全方向に少し膨らませたいときの余白 [同じ単位]

    戻り値:
      indices: np.ndarray, box 内に入る点のインデックス
    """
    P = np.asarray(point_cloud_G)
    x, y, z = P[:, 0], P[:, 1], P[:, 2]

    hx = fingertip_width   / 2.0 + margin
    hy = gripper_aperture  / 2.0 + margin
    hz = fingertip_height  / 2.0 + margin

    mask = (
        (np.abs(x) <= hx) &
        (np.abs(y) <= hy) &
        (np.abs(z) <= hz)
    )

    indices = np.nonzero(mask)[0]
    return indices
