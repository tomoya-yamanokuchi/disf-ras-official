import numpy as np
from domain_object.builder import DomainObject
from value_object import PointNormalUnitPairs, PointNormalIndexUnitPairs


class Ep_Parameters:
    def __init__(self, domain_object: DomainObject):
        pass

    def compute(self,
            source : PointNormalIndexUnitPairs,
            target : PointNormalUnitPairs,
            # ----
            delta_d: float,
            v      : np.ndarray,
        ):
        # ---------------------------------------------
        a_list = []
        b_list = []
        # -----
        for i in range(source.points.shape[0]):
            # -----------------------------------------
            j   = source.finger_indices[i]
            p   = source.points[i]
            q   = target.points[i]
            n_q = target.normals[i]
            # -----------------------------------------
            p_hat = p + (0.5 * ((-1)**j) * v * delta_d)
            # -----------------------------------------
            a_i = np.hstack([np.cross(p_hat, n_q),  n_q])
            # a_i = np.cross(p_hat, n_q)
            # a_i = n_q
            # import ipdb; ipdb.set_trace()

            b_i = ((q - p_hat).T @ n_q)
            # -----------------------------------------
            a_list.append(a_i)
            b_list.append(b_i)
        # -----------------------------------------
        A = np.vstack(a_list)
        b = np.vstack(b_list)
        # import ipdb; ipdb.set_trace()
        # ------
        return A, b

