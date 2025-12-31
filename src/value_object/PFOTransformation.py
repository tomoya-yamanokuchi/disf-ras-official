import numpy as np
from typing import NamedTuple
from service import ExtendedRotation
from .RigidTransformation import RigidTransformation


class PFOTransformation(NamedTuple):
    rotation    : ExtendedRotation
    translation : np.ndarray
    delta_d     : float
   # rigid_T     : RigidTransformation
    T_rigid      : np.ndarray


