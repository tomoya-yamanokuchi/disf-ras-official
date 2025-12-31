from typing import NamedTuple
from value_object import TargetPointNormalIndexPairs
from value_object import PointNormalIndexUnitPairs


class ICPResult(NamedTuple):
    source: PointNormalIndexUnitPairs
    target: TargetPointNormalIndexPairs
    num_correspondences: int
