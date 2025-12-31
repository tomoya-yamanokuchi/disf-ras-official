import numpy as np
from ..error import point2plane_error
from ..error import normal_alignment_error
from value_object import PointNormalUnitPairs
from domain_object.builder import SelfContainedDomainObjectBuilder as DomainObject


class OldPalmPoseLeastSquares:
    def __init__(self, domain_object: DomainObject):
        self.finger_indices = domain_object.finger_indices
        self.gripper_normal = domain_object.v0
        self.alpha          = domain_object.alpha
        # ---
        self.delta_d_est    = None

    def update_delta_d(self, delta_d_est: float):
        assert delta_d_est == 0.0
        # ----
        self.delta_d_est = delta_d_est

    def solve_Rt(self,
            source: PointNormalUnitPairs,
            target: PointNormalUnitPairs,
            Rt    : np.ndarray,
        ):
        # ------------------------------------------------
        source_points  = source.points
        source_normals = source.normals
        # ----
        target_points  = target.points
        target_normals = target.normals
        # # ---------------- point to plane ------------------
        point2plane_errors = point2plane_error(source_points, target_points, target_normals)
        A = []
        b = []
        for i in range(len(source_points)):
            j     = self.finger_indices[i]
            # ------------
            v     = self.gripper_normal
            Rtv   = (Rt @ v)
            # -------------
            p     = source_points[i]
            # ----------------------------------------------------
            p_hat = p + (0.5 * ((-1)**j) * Rtv * self.delta_d_est)
            # p_hat = p + (0.5 * ((-1)**j) * v * self.delta_d_est)
            # import ipdb ; ipdb.set_trace()
            # ----------------------------------------------------
            n = target_normals[i]
            cross_prod = np.cross(p_hat, n)
            A.append(np.hstack([cross_prod, n]))  # 回転と並進
            b.append(-point2plane_errors[i])
        A = np.array(A)
        b = np.array(b)
        # ---------------- point to plane ------------------
        errors_normal = normal_alignment_error(source_normals, target_normals)
        # ----
        An = []
        bn = []
        for i in range(len(source_points)):
            npi = source_normals[i]
            nqi = target_normals[i]
            cross_prod = np.cross(npi, nqi)
            An.append(np.hstack([self.alpha * cross_prod, np.zeros(3)]))  # 回転と並進
            bn.append(-self.alpha * errors_normal[i])
        An = np.array(An)
        bn = np.array(bn)
        # ---------------- point to plane ------------------
        A_tilde = np.concatenate([A, An])
        b_tilde = np.concatenate([b, bn])
        # ----
        x, _, _, _ = np.linalg.lstsq(A_tilde, b_tilde, rcond=None)
        # ---------- extract infomation ---------
        rotvec      = x[:3]
        translation = x[3:]
        # import ipdb ; ipdb.set_trace()
        # ----------------------------------------
        return rotvec, translation


