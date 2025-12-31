from typing import NamedTuple
from .IPFOErrors import IPFOErrors
from .PointNormalUnitPairs import PointNormalUnitPairs
from .PFOTransformation import PFOTransformation
import numpy as np


class IPFOParams(NamedTuple):
    transformation : PFOTransformation
    error          : IPFOErrors
    aligned_source : PointNormalUnitPairs
