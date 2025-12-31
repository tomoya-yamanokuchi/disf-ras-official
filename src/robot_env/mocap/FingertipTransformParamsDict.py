from typing import TypedDict
import numpy as np


class FingertipTransformParamsDict(TypedDict):
    rotation_matrix   : np.ndarray # (3,3)
    translation       : np.ndarray # (3,)
    delta_d           : float
