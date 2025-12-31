import numpy as np
from domain_object.builder import DomainObject
from value_object import PointNormalUnitPairs, PointNormalIndexUnitPairs
from service import skew_symmetric_matrix


class Ea_Parameters:
    def __init__(self, domain_object: DomainObject):
        self.beta  = domain_object.beta
        self.n_app = domain_object.n_app

    def compute(self,
            n_z   : np.ndarray,
        ):
        # ----
        n_app_skew = skew_symmetric_matrix(self.n_app)
        n_z_skew   = skew_symmetric_matrix(n_z)
        # ----
        E = np.dot(n_app_skew, n_z_skew)     # (3, 3)
        G = np.hstack([E, np.zeros((3, 3))]) # (3, 6)
        # ----
        f = - np.dot(n_z_skew, self.n_app)[:, None]  # (3, 1)
        # -----
        return G, f
