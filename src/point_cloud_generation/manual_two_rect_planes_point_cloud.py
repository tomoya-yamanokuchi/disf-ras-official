import numpy as np
from scipy.spatial.transform import Rotation


def manual_two_rect_planes_point_cloud(
    size_y: float,
    size_z: float,
    gap: float,
    rotvec,      # (roll, pitch, yaw) in rad
    center,               # world 座標系での「2枚の中点」
    num_points_y: int = 15,
    num_points_z: int = 15,
):
    """
    2枚の平行な長方形平面からなる点群を作る。
    ローカル座標系の約束:
        x軸: 法線方向（指が向き合う方向）
        y軸: たて
        z軸: よこ
        左平面中心: x = -gap/2
        右平面中心: x = +gap/2

    size_y, size_z : それぞれ y, z 方向の長さ
    gap            : 2枚の平面間の距離
    rotvec         : ()
    center         : 2枚の平面の中点の world 座標

    戻り値:
        points      : (2*N, 3)  両方の平面をまとめた点群
        normals     : (2*N, 3)  各点の法線（左右で反対向き）
        points_L, normals_L : 左平面だけ
        points_R, normals_R : 右平面だけ
    """
    center = np.asarray(center, dtype=float)

    # --- 1. ローカル座標で格子点を作る（x=0 の面を基準に） ---
    ys = np.linspace(-size_y / 2.0, size_y / 2.0, num_points_y)
    zs = np.linspace(-size_z / 2.0, size_z / 2.0, num_points_z)
    yy, zz = np.meshgrid(ys, zs)
    yy = yy.ravel()
    zz = zz.ravel()
    N = yy.size

    # 平面 x=0 上の基準点群
    base_plane_local = np.stack([np.zeros(N), yy, zz], axis=1)  # shape (N, 3)

    # --- 2. 左右の平面中心 (ローカル x 方向に ±gap/2 シフト) ---
    offset = np.array([gap / 2.0, 0.0, 0.0])

    plane_L_local = base_plane_local - offset  # 左: x = -gap/2
    plane_R_local = base_plane_local + offset  # 右: x = +gap/2

    # --- 3. ローカル法線（左:+x, 右:-x） ---
    n_L_local = np.tile(np.array([[-1.0, 0.0, 0.0]]), (N, 1))  #
    n_R_local = np.tile(np.array([[1.0, 0.0, 0.0]]), (N, 1)) #

    # --- 4. 回転 R と並進 center を掛けて world 座標に変換 ---
    # R = Rotation.from_euler("xyz", euler_angle_rad).as_matrix()
    R = Rotation.from_rotvec(rotvec).as_matrix()

    points_L = (R @ plane_L_local.T).T + center
    points_R = (R @ plane_R_local.T).T + center

    normals_L = (R @ n_L_local.T).T
    normals_R = (R @ n_R_local.T).T

    # --- 5. 2枚をまとめた object surface ---
    points  = np.vstack([points_L,  points_R])
    normals = np.vstack([normals_L, normals_R])

    return points, normals, points_L, normals_L, points_R, normals_R
