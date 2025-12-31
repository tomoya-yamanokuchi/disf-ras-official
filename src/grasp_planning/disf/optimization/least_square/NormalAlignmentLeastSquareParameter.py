import numpy as np
from value_object import PointNormalUnitPairs, PointNormalIndexUnitPairs
from domain_object.builder import SelfContainedDomainObjectBuilder as DomainObject


class NormalAlignmentLeastSquareParameter:
    def __init__(self, domain_object: DomainObject):
        self.finger_indices = domain_object.finger_indices
        self.alpha          = domain_object.alpha


    def compute(self,
            source : PointNormalIndexUnitPairs,
            target : PointNormalUnitPairs,
        ):
        # ----------
        num_data = source.points.shape[0]
        # ----------
        An = []
        bn = []
        # ----------
        for i in range(num_data):
            n_p = source.normals[i]
            n_q = target.normals[i]
            # ----------
            An.append(  self.alpha * np.cross(n_p, n_q))
            bn.append(- self.alpha * (np.dot(n_p, n_q) + 1))
        # ----------
        An = np.array(An)
        bn = np.array(bn)
        # ---------------------------------------
        return (An, bn)



