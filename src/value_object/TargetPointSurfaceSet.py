from typing import TypedDict, NamedTuple
import numpy as np
from .TargetPointNormalIndexPairs import TargetPointNormalIndexPairs
from .PointNormalIndexUnitPairs import PointNormalIndexUnitPairs

class TargetPointSurfaceSet(NamedTuple):
    correspondence  : TargetPointNormalIndexPairs
    contact_surface : PointNormalIndexUnitPairs
    whole_surface   : PointNormalIndexUnitPairs
