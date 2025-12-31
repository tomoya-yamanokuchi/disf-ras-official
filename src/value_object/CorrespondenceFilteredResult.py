from typing import TypedDict, NamedTuple
import numpy as np
from .TargetPointNormalIndexPairs import TargetPointNormalIndexPairs
from .PointNormalIndexUnitPairs import PointNormalIndexUnitPairs


class CorrespondenceFilteredResult(NamedTuple):
    filtered_target                     : TargetPointNormalIndexPairs
    filtered_source                     : PointNormalIndexUnitPairs
    # unique_corres_indices_first_position: np.ndarray


