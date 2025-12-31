import numpy as np
from domain_object.builder import SelfContainedDomainObjectBuilder as DomainObject
from value_object import PointNormalUnitPairs


class FingertipDisplacementLeastSquare:
    def __init__(self, domain_object: DomainObject):
        self.finger_indices = domain_object.finger_indices
        self.alpha          = domain_object.alpha
        self.d_min          = domain_object.d_min
        self.d_max          = domain_object.d_max
        self.d0             = domain_object.d0

    def solve_delta_d(self,
            source : PointNormalUnitPairs,
            target : PointNormalUnitPairs,
            R_opt  : np.ndarray,
            t_opt  : np.ndarray,
            v      : np.ndarray,
        ):
        # ----------------------------------
        source_points  = source.points  # p
        source_normals = source.normals # np
        # --
        target_points  = target.points  # q
        target_normals = target.normals # nq
        # -----------------------------------
        a = []
        b = []
        for i in range(source_points.shape[0]):
            # -----------------------------------
            j    = self.finger_indices[i]
            pij  = source_points[i]
            qij  = target_points[i]
            nqij = target_normals[i]
            # -----------
            Rv   = (R_opt @ v)
            # -----------------------------------
            aij  = 0.5 * ((-1) ** (j-1)) * (Rv.T @ nqij)
            bij  = ((R_opt @ pij) + t_opt - qij).T @ nqij
            # -----------------------------------
            a.append(aij)
            b.append(bij)
        # -----------------------------------
        a = np.array(a)
        b = np.array(b)
        # -----------------------------------
        delta_d_hat  = np.sum(a * b) / np.sum(a ** 2)
        delta_d_star = np.clip(delta_d_hat, self.d_min - self.d0, self.d_max - self.d0)
        # -------------------
        return delta_d_star