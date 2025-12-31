import numpy as np
from typing import NamedTuple
from service import ExtendedRotation
from .RigidTransformation import RigidTransformation


class PFOResult(NamedTuple):
    rotation    : ExtendedRotation
    translation : np.ndarray
    delta_d     : float
    T_rigid     : np.ndarray


