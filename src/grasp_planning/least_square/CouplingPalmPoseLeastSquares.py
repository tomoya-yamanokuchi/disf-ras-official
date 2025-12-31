import numpy as np
from value_object import PointNormalUnitPairs
from domain_object.builder import SelfContainedDomainObjectBuilder as DomainObject


class CouplingPalmPoseLeastSquares:
    def __init__(self, domain_object: DomainObject):
        self.finger_indices = domain_object.finger_indices
        self.alpha          = domain_object.alpha

    def solve_Rt(self,
            source0    : PointNormalUnitPairs,
            target     : PointNormalUnitPairs,
            delta_d_opt: float,
            v          : np.ndarray,
        ):
        # ------------------------------------------------
        source_points  = source0.points
        source_normals = source0.normals
        # ----
        target_points  = target.points
        target_normals = target.normals
        # --------------------------------------------------------
        #                   point to plane
        # --------------------------------------------------------
        A = []
        b = []
        for i in range(len(source_points)):
            # -----------------------------------------
            j  = self.finger_indices[i]
            p  = source_points[i]
            q  = target_points[i]
            nq = target_normals[i]
            # -----------------------------------------
            p_hat = p + (0.5 * ((-1)**j) * v * delta_d_opt)
            # -----------------------------------------
            bij = (q - p_hat).T @ nq
            # -----------------------------------------
            A.append(np.hstack([np.cross(p_hat, nq), nq]))
            b.append(bij)
            # -----------------------------------------
        A = np.array(A)
        b = np.array(b)

        # --------------------------------------------------------
        #                  normal alignment
        # --------------------------------------------------------
        An = []
        bn = []
        for i in range(len(source_points)):
            # ------------------------------
            npi = source_normals[i]
            nqi = target_normals[i]
            # ------------------------------
            cross_prod = np.cross(npi, nqi)
            # ------------------------------
            bij = - self.alpha * (npi.T @ nqi + 1)
            # ------------------------------
            An.append(np.hstack([self.alpha * cross_prod, np.zeros(3)]))  # 回転と並進
            bn.append(bij)
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


