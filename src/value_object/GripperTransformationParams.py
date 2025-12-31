from typing import TypedDict
import numpy as np
# from service import ExtendedRotation



class GripperTransformationParams(TypedDict):
    rotation_matrix : np.ndarray # approximated rotation_matrix with (I + skew-symmetrix(r))
    translation     : np.ndarray
    delta_d         : float
    v               : np.ndarray