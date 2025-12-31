import numpy as np
from value_object import PointNormalUnitPairs, PointNormalIndexUnitPairs
from domain_object.builder import SelfContainedDomainObjectBuilder as DomainObject
from .Ea_Parameters import Ea_Parameters
from typing import Tuple

class PalmLeastSquare_Ea_Parameter:
    def __init__(self, domain_object: DomainObject):
        self.Ea_params = Ea_Parameters(domain_object)

    def compute(self,
            n_z: np.ndarray,
        ):
        # --------------------------------------------------------
        G, f = self.Ea_params.compute(n_z)
        # ---------------------
        return G, f



