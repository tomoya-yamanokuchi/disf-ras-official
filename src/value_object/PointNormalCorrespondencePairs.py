from typing import TypedDict, NamedTuple
from .PointNormalUnitPairs import PointNormalUnitPairs


class PointNormalCorrespondencePairs(NamedTuple):
    source : PointNormalUnitPairs
    target : PointNormalUnitPairs