from typing import TypedDict, NamedTuple
import numpy as np



class IPFOErrors(NamedTuple):
    total             : np.ndarray
    point2plaine      : np.ndarray
    normal_alignment  : np.ndarray
    approach_alignment: np.ndarray
