import os
import numpy as np
from value_object import IPFOErrors
from domain_object.builder import DomainObject
from value_object import SourcePointSurfaceSet
from value_object import TargetPointSurfaceSet

class GeomError:
    def __init__(self, domain_object: DomainObject):
        self.name = "GeomError"
        # ---
        self.Ep = domain_object.Ep
        self.En = domain_object.En

    def compute(self,
            source_set        : SourcePointSurfaceSet,
            target_set        : TargetPointSurfaceSet,
        ) -> IPFOErrors:
        # -------
        Ep         = self.Ep.compute(source_set, target_set, history_append=False)
        beta_En    = self.En.compute_with_weight(source_set, target_set, history_append=False)
        # -------
        geom_error = (Ep + beta_En)
        # -------
        return geom_error
