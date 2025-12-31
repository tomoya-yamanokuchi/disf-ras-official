import numpy as np
from value_object import PointNormalUnitPairs
from domain_object.builder import SelfContainedDomainObjectBuilder as DomainObject
from .NormalAlignmentLeastSquareParameter import NormalAlignmentLeastSquareParameter
from .ApproachDirectionLeastSquareParameter import ApproachDirectionLeastSquareParameter


class PalmRotationLeastSquare_EnEa:
    def __init__(self, domain_object: DomainObject):
        self.ls_En = NormalAlignmentLeastSquareParameter(domain_object)
        self.ls_Ea = ApproachDirectionLeastSquareParameter(domain_object)

    def solve_R(self,
            source : PointNormalUnitPairs,
            target : PointNormalUnitPairs,
            n_z    : np.ndarray,
        ):
        (An, bn) = self.ls_En.compute(source, target)
        (Aa, ba) = self.ls_Ea.compute(n_z)
        # -----
        # A_tilde = Aa
        # b_tilde = ba
        # ------
        A_tilde = np.concatenate([An, Aa])
        b_tilde = np.concatenate([bn, ba])
        # ---------------------------------------
        x, _, _, _ = np.linalg.lstsq(A_tilde, b_tilde, rcond=None)
        rotvec     = x
        # ----------------------------------------
        # import ipdb ; ipdb.set_trace()
        return rotvec



