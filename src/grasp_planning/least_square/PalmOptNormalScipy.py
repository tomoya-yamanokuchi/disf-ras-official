import numpy as np
from ..error import point2plane_error
from ..error import normal_alignment_error
from scipy.optimize import least_squares
from service import small_angle_rotation_matix
from value_object import PointNormalUnitPairs
from domain_object.builder import SelfContainedDomainObjectBuilder as DomainObject


def loss_function(source_normals, target_normals, rotvec):

    R_skew = small_angle_rotation_matix(r=rotvec)
    Rn     = np.dot(source_normals, R_skew.T)

    loss = (np.sum(Rn * target_normals, axis=1) + 1)

    return np.array(loss)


class PalmOptNormalScipy:
    def __init__(self, domain_object: DomainObject):
        self.finger_indices = domain_object.finger_indices
        self.gripper_normal = domain_object.v0
        self.alpha          = domain_object.alpha
        # ---
        self.delta_d_est    = None

    def update_delta_d(self, delta_d_est: float):
        self.delta_d_est = delta_d_est

    def solve_Rt(self, source: PointNormalUnitPairs, target: PointNormalUnitPairs):
        # ------------------------------------------------
        source_points  = source.points
        source_normals = source.normals
        # ----
        target_points  = target.points
        target_normals = target.normals
        # ----
        # An = []
        # bn = []
        # for i in range(len(source_points)):
        #     npi = source_normals[i]
        #     nqi = target_normals[i]
        #     # ---
        #     cross_prod = np.cross(npi, nqi)
        #     ai         = np.hstack([self.alpha * cross_prod, np.zeros(3)])
        #     An.append(ai)  # 回転と並進
        #     # ---
        #     bi = - self.alpha * (npi.T @ nqi) - self.alpha
        #     bn.append(bi)
        #     break
        # # import ipdb ; ipdb.set_trace()
        # An = np.array(An)
        # bn = np.array(bn)
        # ---------------------------------------
        loss   = lambda x: loss_function(source_normals, target_normals, x)
        result = least_squares(loss, np.zeros(3))
        # import ipdb ; ipdb.set_trace()
        # ---------- extract infomation ---------
        rotvec      = result.x
        translation = np.zeros(3)

        # ----------------------------------------
        # import ipdb ; ipdb.set_trace()
        return rotvec, translation


