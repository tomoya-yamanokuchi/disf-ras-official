import numpy as np
from value_object import PointNormalUnitPairs
from value_object import PointNormalIndexUnitPairs
from domain_object.builder import SelfContainedDomainObjectBuilder as DomainObject


class FingertipDisplacement:
    def __init__(self, domain_object: DomainObject):
        self.d_min          = domain_object.d_min
        self.d_max          = domain_object.d_max

    def solve_delta_d(self,
            source : PointNormalIndexUnitPairs,
            target : PointNormalUnitPairs,
            R      : np.ndarray,
            t      : np.ndarray,
            v      : np.ndarray,
            d      : np.ndarray,
        ):
        # -------
        K_list = []
        H_list = []
        # -------
        for i in range(source.points.shape[0]):
            # -----------------------------------
            j   = source.finger_indices[i]
            p   = source.points[i]
            q   = target.points[i]
            n_q = target.normals[i]
            # -----------------------------------
            K_ij  = 0.5 * ((-1) ** (j-1)) * ((R @ v).T @ n_q)
            H_ij  = ((R @ p) + t - q).T @ n_q
            # -----------------------------------
            K_list.append(K_ij)
            H_list.append(H_ij)
        # -----------------------------------
        K = np.vstack(K_list)
        H = np.vstack(H_list)
        # -----------------------------------
        delta_d_hat  = np.sum(K * H) / np.sum(K ** 2)
        delta_d_star = np.clip(delta_d_hat, self.d_min - d, self.d_max - d)
        # -------------------
        # import ipdb ; ipdb.set_trace()
        # print("-----------------------------")
        # print("delta_d_star = ", delta_d_star)
        # print("-----------------------------")
        return delta_d_star


