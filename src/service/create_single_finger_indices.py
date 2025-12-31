import numpy as np


def create_single_finger_indices(num_fingertip_surface_points: int, j:int):
    """
        We assume that source_points are created by concatenation
        under a specific order, ex. from right to left:
            --> source_points = np.concatenat([right_source_points, left_source_points])
    """
    # ---
    return np.tile(np.array([j]), (num_fingertip_surface_points,)) # j=1