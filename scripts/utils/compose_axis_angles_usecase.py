import math
import numpy as np
from service import compose_axis_angles
from service import ExtendedRotation

# import ipdb; ipdb.set_trace()
def run(axes, angles):
    axis_total, angle_total = compose_axis_angles(axes, angles)
    rotvec = (axis_total * angle_total)

    print("-----------------------------------------------")
    print("quat =", np.array2string(ExtendedRotation.from_rotvec(rotvec).as_quat_scalar_first(), separator=', '))
    print("-----------------------------------------------")
    print("rotvec [rad] =", np.array2string(rotvec, separator=', '))
    print("rotvec [deg] =", np.array2string(np.rad2deg(rotvec), separator=', '))
    print("-----------------------------------------------")
    print("axis   =", axis_total)
    print("angle  =", angle_total, "[rad]")
    print("angle_deg =", np.rad2deg(angle_total), "[deg]")
    print("-----------------------------------------------")


if __name__ == '__main__':
    # ==================== case3 ====================
    axes = [
        (0.0, 0.0, 1.0),  # z軸
        (1.0, 0.0, 0.0),  # x軸
    ]
    angles = [
        math.pi / 2,  # z軸まわり π/2
        -math.pi / 4,  # x軸まわり π/4
    ]

    # ==================== case3 ====================
    axes = [
        (0.0, 0.0, 1.0),  # z軸
        # (1.0, 0.0, 0.0),  # x軸
    ]
    angles = [
        np.deg2rad(45), # z軸まわり π/2
        # -math.pi / 4,  # x軸まわり π/4
    ]

    '''
                YCB Object
    '''
    # ==================== 006_mustard_bottle ====================
    axes = [
        (1.0, 0.0, 0.0),
        (0, 1, 0),
        (0, 0, 1),
    ]
    angles = [
        np.deg2rad(-90),
        np.deg2rad(90),
        np.deg2rad(-45),
    ]

    # # ==================== 001_chips_can ====================
    # axes = [
    #     (0, 0, 1),  # y軸
    # ]
    # angles = [
    #     np.deg2rad(45), # z軸まわり π/2
    # ]

    axes = [
        (0, 0, 1),
        (1, 0, 1),
    ]
    angles = [
        np.deg2rad(90), # z軸まわり π/2
        np.deg2rad(90), # z軸まわり π/2
    ]

    # # ==================== 033_spatula ====================
    # axes = [
    #     (0, 1, 0),  # y軸
    # ]
    # angles = [
    #     np.deg2rad(90), # z軸まわり π/2
    # ]

    # # ==================== test ====================
    # axes = [
    #     (0.0, 0.0, 1.0),
    #     (0.0, 1.0, 0.0),
    # ]
    # angles = [
    #     np.deg2rad(-45), # z軸まわり π/2
    #     np.deg2rad(45),  # x軸まわり π/4
    # ]

    # axes = [
    #     (0, 1, 0),
    # ]
    # angles = [
    #     np.deg2rad(90), # z軸まわり π/2
    #     # np.deg2rad(55), # z軸まわり π/2
    # ]

    axes = [
        (1, 0, 0),
        # (0, 0, 1),
    ]
    angles = [
        np.deg2rad(90), # z軸まわり π/2
        # np.deg2rad(-90), # z軸まわり π/2
    ]


    axes = [
        (0, 1, 0),
        (0, 0, 1),
    ]
    angles = [
        np.deg2rad(90), # z軸まわり π/2
        np.deg2rad(45), # z軸まわり π/2
    ]


    run(axes, angles)
