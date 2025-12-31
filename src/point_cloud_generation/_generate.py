import numpy as np
from .manual_two_rect_planes_point_cloud import manual_two_rect_planes_point_cloud


# ==================== Translation ====================
# ------------------------  case1  ----------------------------
center = (0.015, 0.015, -0.015)
# rotvec = [0.        , 0.        , 0]
rotvec = [0.        , 0.        , 1.57079633]

# # # ------------------------  case2  ----------------------------
# center = (0.045, 0.015, -0.015)
# rotvec = [0.        , 0.        , 1.57079633]
# # # # ------------------------  case3  ----------------------------
# # center = (0.03, 0.015, -0.015)
# center = (0.042426406871192854, 0, -0.015)
# rotvec = [0, 0,  np.deg2rad(45)]

# # ------------------------  case4  ----------------------------
# center = (0.045, 0.045,  -0.015)
# rotvec = [0, 0,  np.deg2rad(45)]

# # ------------------------  case4  ----------------------------
# center = (0.045, 0.045,  -0.015)
# rotvec = [0, 0,  np.deg2rad(45)]

# import ipdb; ipdb.set_trace()
# # ==================== Rotation ====================
# # ------------------------  case4  ----------------------------
# euler_angle_rad    = (0, 0.0, np.deg2rad(0))
# center             = (0.015, 0.015, 0.015)
# # ------------------------  case5  ----------------------------
# euler_angle_rad    = (0, 0.0, np.deg2rad(0))
# center             = (0.045, 0.015, 0.015)
# # ------------------------  case6  ----------------------------
# euler_angle_rad    = (0, 0.0, np.deg2rad(0))
# center             = (0.015, 0.045, 0.015)

# # ==================== Trans & Rotation ====================
# # ------------------------  case7  ----------------------------
# euler_angle_rad    = (0, 0.0, np.deg2rad(0))
# center             = (0.015, 0.015, 0.045)
# # ------------------------  case8  ----------------------------
# euler_angle_rad    = (0, 0.0, np.deg2rad(0))
# center             = (0.045, 0.015, 0.045)
# # ------------------------  case9  ----------------------------
# euler_angle_rad    = (0, 0.0, np.deg2rad(0))
# center             = (0.015, 0.045, 0.045)

# # ------------------------  case10  ----------------------------
# euler_angle_rad    = (0, 0.0, np.deg2rad(45))
# center             = (0.045, 0.045, 0.045)



def _generate():
    points, normals, points_L, normals_L, points_R, normals_R = \
    manual_two_rect_planes_point_cloud(
        size_y         = 0.03,    # たて
        size_z         = 0.03,    # よこ
        gap            = 0.03,    # 2枚の距離（短径）
        rotvec         = rotvec,
        center         = center,  # 2枚の中点
        num_points_y   = 15,
        num_points_z   = 15,
    )

    return points, normals


if __name__ == '__main__':
    _generate()
