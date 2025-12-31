import numpy as np
from typing import NamedTuple
from service.ExtendedRotation import ExtendedRotation
# from .PointNormalUnitPairs import PointNormalUnitPairs
from .IPFOErrors import IPFOErrors

class ISFResult(NamedTuple):
    rotation        : ExtendedRotation
    translation     : np.ndarray
    delta_d         : np.ndarray
    # ----
    error           : IPFOErrors
    # aligned_source  : PointNormalUnitPairs
    # ----
    Rt_hist       : np.ndarray
    R_object      : np.ndarray
    pos_object    : np.ndarray
    pfo_error_hist: np.ndarray
    es_hist       : np.ndarray
