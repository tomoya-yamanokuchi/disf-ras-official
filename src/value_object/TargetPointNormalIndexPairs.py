from typing import NamedTuple
import numpy as np


class TargetPointNormalIndexPairs(NamedTuple):
    points  : np.ndarray
    normals : np.ndarray
    indices : np.ndarray
