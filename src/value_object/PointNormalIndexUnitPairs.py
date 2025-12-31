from typing import TypedDict, NamedTuple
import numpy as np


class PointNormalIndexUnitPairs(NamedTuple):
    points        : np.ndarray
    normals       : np.ndarray
    finger_indices: np.ndarray