import numpy as np




def generate_surface(
        distance_between_points : float,
        gripper_z_width         : float,
        gripper_x_width         : float,
        d0                      : float,
        finger_index            : int,
        robot_name              : str,
    ):

    # if robot_name == "panda":
    #     from .generate_surface_panda_hand import generate_surface_panda_hand
    #     return generate_surface_panda_hand(
    #         num_fingertip_surface_points,
    #         d0, finger_index)

    # if robot_name == "ur5e":
    from .generate_surface_ur5e_hand import generate_surface_ur5e_hand
    return generate_surface_ur5e_hand(
        distance_between_points = distance_between_points, # [m]
        gripper_z_width         = gripper_z_width, # [m]
        gripper_x_width         = gripper_x_width, # [m]
        d0                      = d0, # [m]
        finger_index            = finger_index,
    )
