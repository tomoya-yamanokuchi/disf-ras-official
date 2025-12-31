from typing import TypedDict, NamedTuple
import numpy as np


class PointNormalUnitPairs(NamedTuple):
    points        : np.ndarray
    normals       : np.ndarray