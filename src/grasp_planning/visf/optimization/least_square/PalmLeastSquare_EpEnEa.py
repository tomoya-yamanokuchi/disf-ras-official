import numpy as np
from value_object import PointNormalUnitPairs
from domain_object.builder import SelfContainedDomainObjectBuilder as DomainObject
from .PalmLeastSquare_EpEn_Parameter import PalmLeastSquare_EpEn_Parameter
from .PalmLeastSquare_Ea_Parameter import PalmLeastSquare_Ea_Parameter


class PalmLeastSquare_EpEnEa:
    def __init__(self, domain_object: DomainObject):
        self.beta    = domain_object.beta
        self.ls_EpEn = PalmLeastSquare_EpEn_Parameter(domain_object)
        self.ls_Ea   = PalmLeastSquare_Ea_Parameter(domain_object)


    def solve_Rt(self,
            source : PointNormalUnitPairs,
            target : PointNormalUnitPairs,
            delta_d: float,
            v      : np.ndarray,
            n_z    : np.ndarray,
        ):
        # ----------------------------------------------------
        A, b = self.ls_EpEn.compute(source, target, delta_d, v) # A: (12, 6)
        G, f = self.ls_Ea.compute(n_z)
        # ----------------------------------------------------
        AT_A = A.T @ A
        GT_G = G.T @ G
        AT_b = A.T @ b
        GT_f = G.T @ f
        # -----------------------------------------------------
        AA_GG = AT_A + (self.beta * GT_G)  # (6, 6)
        Ab_Gf = AT_b + (self.beta * GT_f)  # (6, 1)
        # -----------------------------------------------------
        x = np.linalg.pinv(AA_GG) @ Ab_Gf
        x : np.ndarray = x.squeeze(1)
        # -----------------------------------------------------
        rotvec      = x[:3]
        translation = x[3:]
        # import ipdb ; ipdb.set_trace()
        # -----------------------------------------------------
        return rotvec, translation



