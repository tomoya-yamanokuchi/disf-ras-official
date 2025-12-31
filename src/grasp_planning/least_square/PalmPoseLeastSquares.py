import numpy as np
from ..error import point2plane_error
from ..error import normal_alignment_error
from value_object import PointNormalUnitPairs
from domain_object.builder import SelfContainedDomainObjectBuilder as DomainObject


class PalmPoseLeastSquares:
    def __init__(self, domain_object: DomainObject):
        self.alpha = domain_object.alpha

    def solve_Rt(self,
            source: PointNormalUnitPairs,
            target: PointNormalUnitPairs,
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
            # -------------
            p = source_points[i]
            n = target_normals[i]
            cross_prod = np.cross(p, n)
            # -------------
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
        # ----------------------------------------
        return rotvec, translation


