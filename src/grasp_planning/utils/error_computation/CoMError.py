import os
import numpy as np
from value_object import IPFOErrors
from domain_object.builder import DomainObject
from value_object import SourcePointSurfaceSet
from value_object import TargetPointSurfaceSet
from service import calculate_centroid

class CoMError:
    def __init__(self, domain_object: DomainObject):
        self.name = "CoMError"

    def compute(self,
            source_set        : SourcePointSurfaceSet,
            target_set        : TargetPointSurfaceSet,
        ) -> IPFOErrors:
        # -------
        c_source = calculate_centroid(point_clouds=source_set.correspondence.points)
        c_target = calculate_centroid(point_clouds=target_set.correspondence.points)
        # -------
        com_error = np.linalg.norm(c_target - c_source)
        # -------
        return com_error
