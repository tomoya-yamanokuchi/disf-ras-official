import numpy as np
from domain_object.builder import SelfContainedDomainObjectBuilder as DomainObject
from service import skew_symmetric_matrix


class PalmPoseApproachRLestSquare:
    def __init__(self, domain_object: DomainObject):
        self.na = domain_object.n_app

    def solve_R(self, nz: np.ndarray):
        # Ensure n_z and n_app are numpy arrays
        nz      = np.asarray(nz, dtype=float)
        na      = np.asarray(self.na, dtype=float)
        # ------
        na_skew = skew_symmetric_matrix(r=na)
        nz_skew = skew_symmetric_matrix(r=nz)
        E       = np.dot(na_skew, nz_skew)
        f       = - np.dot(nz_skew, na)
        # ------
        delta_r, _, _, _ = np.linalg.lstsq(E, f, rcond=None)
        # ------
        """
        初期値によって回転角度が0になってしまう
            (1) 平行: nz x na = 0 -> f = - (nz x na) = 0
            (2) 直交: E -> low rank
        """
        return delta_r



