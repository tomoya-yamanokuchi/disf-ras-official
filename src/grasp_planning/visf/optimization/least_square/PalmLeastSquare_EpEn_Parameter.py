import numpy as np
from value_object import PointNormalUnitPairs, PointNormalIndexUnitPairs
from domain_object.builder import SelfContainedDomainObjectBuilder as DomainObject
from .Ep_Parameters import Ep_Parameters
from .En_Parameters import En_Parameters


class PalmLeastSquare_EpEn_Parameter:
    def __init__(self, domain_object: DomainObject):
        self.Ep_params = Ep_Parameters(domain_object)
        self.En_params = En_Parameters(domain_object)


    def compute(self,
            source : PointNormalIndexUnitPairs,
            target : PointNormalUnitPairs,
            delta_d: float,
            v      : np.ndarray,
        ):
        # --------------------------------------------------------
        A,  b  = self.Ep_params.compute(source, target, delta_d, v)
        An, bn = self.En_params.compute(source, target)
        # ---
        # cond_A  = np.linalg.cond(A)
        # cond_An = np.linalg.cond(An)
        # ---
        A_tilde = np.vstack([A, An])
        b_tilde = np.vstack([b, bn])
        # ---------------------
        # x, _, _, _  = np.linalg.lstsq(A_tilde, b_tilde, rcond=None)
        # rotvec      = x[:3]; print("rotvec = ", rotvec)
        # translation = x[3:]; print("translation = ", translation)
        # import ipdb; ipdb.set_trace()
        return A_tilde, b_tilde


