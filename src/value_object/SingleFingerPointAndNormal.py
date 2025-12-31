from typing import TypedDict
import numpy as np

class SingleFingerPointAndNormal(TypedDict):
    pj  : np.ndarray
    qj  : np.ndarray
    npj : np.ndarray
    nqj : np.ndarray