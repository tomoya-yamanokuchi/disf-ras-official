from typing import TypedDict, NamedTuple
import numpy as np
from value_object import PointNormalUnitPairs
from value_object import PointNormalIndexUnitPairs


class ICPFiltering(NamedTuple):
    selected_indices: np.ndarray
    distances       : np.ndarray
    # ---
    source          : PointNormalIndexUnitPairs
    target          : PointNormalUnitPairs
