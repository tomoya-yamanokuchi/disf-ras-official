import numpy as np
from domain_object.builder import DomainObject
from value_object import PointNormalUnitPairs
from value_object import PointNormalIndexUnitPairs


class Point2PlaneLeastSquareWithFingertipDisplacement:
    def __init__(self, domain_object: DomainObject):
        self.d_min          = domain_object.d_min
        self.d_max          = domain_object.d_max

    def solve_delta_d(self,
            source : PointNormalIndexUnitPairs,
            target : PointNormalUnitPairs,
            v      : np.ndarray,
            d      : np.ndarray,
        ):
        # ------
        a = []
        b = []
        # -------
        for i in range(source.points.shape[0]):
            # -----------------------------------
            j    = source.finger_indices[i]
            p    = source.points[i]
            q    = target.points[i]
            n_q  = target.normals[i]
            # -----------------------------------
            aij  = 0.5 * ((-1) ** (j-1)) * np.dot(v, n_q)
            bij  = np.dot((p - q), n_q)
            # -----------------------------------
            a.append(aij)
            b.append(bij)
        # -----------------------------------
        a = np.array(a)
        b = np.array(b)
        # -----------------------------------
        delta_d_hat  = np.sum(a * b) / np.sum(a ** 2)
        # ---
        delta_d_star = np.clip(
            a     = delta_d_hat,
            a_min = self.d_min - d,
            a_max = self.d_max - d,
        )
        # import ipdb ; ipdb.set_trace()
        # -------------------
        return delta_d_star