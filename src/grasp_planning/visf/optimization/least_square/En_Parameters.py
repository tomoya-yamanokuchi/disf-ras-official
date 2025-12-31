import numpy as np
from domain_object.builder import DomainObject
from value_object import PointNormalUnitPairs, PointNormalIndexUnitPairs


class En_Parameters:
    def __init__(self, domain_object: DomainObject):
        self.alpha = domain_object.alpha

    def compute(self,
            source : PointNormalIndexUnitPairs,
            target : PointNormalUnitPairs,
        ):
        # ---------------------------------------------
        an_list = []
        bn_list = []
        # ------
        for i in range(source.points.shape[0]):
            # ------------------------------
            n_p = source.normals[i]
            n_q = target.normals[i]
            # ------------------------------
            a_i = np.hstack([self.alpha*np.cross(n_p, n_q), np.zeros(3)])
            b_i = (- self.alpha * (n_p.T @ n_q + 1))
            # ------------------------------
            an_list.append(a_i)
            bn_list.append(b_i)
        # -----------------------------------------
        An = np.vstack(an_list)
        bn = np.vstack(bn_list)
        # -----
        return An, bn
