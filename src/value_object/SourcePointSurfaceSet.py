from typing import TypedDict, NamedTuple
from .PointNormalIndexUnitPairs import PointNormalIndexUnitPairs

class SourcePointSurfaceSet(NamedTuple):
    correspondence : PointNormalIndexUnitPairs
    surface        : PointNormalIndexUnitPairs