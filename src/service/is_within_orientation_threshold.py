import numpy as np


def is_within_orientation_threshold(
        error_quat: np.ndarray,
        threshold : float,
    ):
    """
    Checks if the rotation error (represented as a quaternion) is within the threshold.

    Args:
        error_quat (numpy.ndarray): The quaternion representing the rotation error [w, x, y, z].
        threshold (float): The maximum allowable rotation error in radians.

    Returns:
        bool: True if within the threshold, False otherwise.
    """
    # Compute the rotation angle from the quaternion
    w              = error_quat[0]
    rotation_angle = 2 * np.arccos(np.clip(abs(w), -1.0, 1.0))  # Clip to avoid numerical instability

    # Check if the angle is within the threshold
    # import ipdb; ipdb.set_trace()
    return rotation_angle < threshold

