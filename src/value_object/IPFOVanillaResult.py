from typing import NamedTuple
import numpy as np
from value_object import SourcePointSurfaceSet
from value_object import IPFOErrors

class IPFOVanillaResult(NamedTuple):
    R_opt                 : np.ndarray
    t_opt                 : np.ndarray
    delta_d_opt           : np.ndarray
    e_opt                 : IPFOErrors
    aligned_source_set    : SourcePointSurfaceSet
    # aligned_source_surface: PointNormalIndexUnitPairs
