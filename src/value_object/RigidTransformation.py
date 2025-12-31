import numpy as np
from typing import NamedTuple


class RigidTransformation(NamedTuple):
    T_rigid     : np.ndarray
    T_base      : np.ndarray
    T_aux       : np.ndarray
    t_aux       : np.ndarray
