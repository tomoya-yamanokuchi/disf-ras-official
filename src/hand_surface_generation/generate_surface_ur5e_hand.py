import numpy as np


def get_axis_points(gripper_width, distance_between_points):
    # ---
    axis_points = np.hstack([
    np.arange(start=0.0, stop=gripper_width, step=distance_between_points),
    np.array([gripper_width])])
    # ---
    return axis_points


def generate_surface_ur5e_hand(
        distance_between_points     : float, # [m]
        gripper_z_width             : float, # [m]
        gripper_x_width             : float, # [m]
        d0                          : float, # [m]
        finger_index                : int,
    ):
    # ----
    assert finger_index in [1, 2]
    # --------- get axis points  ---------
    z_axis_points = get_axis_points(gripper_z_width, distance_between_points)
    x_axis_points = get_axis_points(gripper_x_width, distance_between_points)
    # ---------- create surafce ----------
    points = []
    for k in range(len(x_axis_points)):
        for m in range(len(z_axis_points)):
            point = np.array([x_axis_points[k], 0, z_axis_points[m]])
            points.append(point)
    # ------ shift surafce to origin ------
    y_finger_offset = ((-1)**finger_index) * (0.5 * d0)

    offset = np.array([gripper_x_width*(-0.5), y_finger_offset, gripper_z_width*(-0.5)])
    points = (points + offset)

    return np.array(points)


if __name__ == '__main__':
    surface = generate_surface_ur5e_hand(
        distance_between_points = 0.001,
        gripper_z_width         = 0.06,
        gripper_x_width         = 0.021,
        d0                      = 0.050,
        finger_index            = 1,
    )

    print(f"\n surface.shape = {surface.shape}\n")
    print(surface)

    '''
    メートル表記の場合の対応表
        A.BCDE [m]

        1 [m] -> 100 [cm]
        1 [cm] -> 10 [mm]
        1 [m] -> 1000[mm]
        1 [mm] -> 0.001[m]
        20 [mm] -> 0.02[m]
    '''
