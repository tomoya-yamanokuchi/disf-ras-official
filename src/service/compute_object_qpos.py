import numpy as np
from .ExtendedRotation import ExtendedRotation

def compute_object_qpos(translation: np.ndarray, rotvec_degree: np.ndarray):
    rotvec_rad  = np.deg2rad(rotvec_degree)
    rotation    = ExtendedRotation.from_rotvec(rotvec_rad)
    quat        = rotation.as_quat_scalar_first()
    qpos_object = np.hstack([translation, quat])
    return qpos_object
