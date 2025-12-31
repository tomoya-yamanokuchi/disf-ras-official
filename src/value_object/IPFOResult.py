from typing import NamedTuple
import numpy as np
from value_object import PointNormalIndexUnitPairs
from value_object import SourcePointSurfaceSet
from .IPFOErrors import IPFOErrors

class IPFOResult(NamedTuple):
    R_sum                 : np.ndarray
    t_sum                 : np.ndarray
    delta_d_sum           : np.ndarray
    d                     : float
    e_geom                : float
    e_com                 : float
    e_p_sum               : IPFOErrors #np.ndarray
    # aligned_source        : PointNormalIndexUnitPairs
    # aligned_source_surface: PointNormalIndexUnitPairs
    aligned_source_set    : SourcePointSurfaceSet
    aligned_n_z           : np.ndarray
    elapsed_time          : np.ndarray
