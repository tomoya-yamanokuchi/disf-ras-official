import numpy as np
from service import ExtendedRotation
from value_object import SourcePointSurfaceSet, TargetPointSurfaceSet
from domain_object.builder import DomainObject
from print_color import print
from service import transform_source_set


class VISF_PalmOptimization:
    def __init__(self, domain_object: DomainObject):
        self.visf_palm_Rt_ls    = domain_object.visf_palm_Rt_ls
        # -----
        self.isf_visualizer     = domain_object.isf_visualizer
        self.text_logger        = domain_object.text_logger
        self.En                 = domain_object.En
        self.Ea                 = domain_object.Ea
        self.verbose            = domain_object.verbose
        self.v0                 = domain_object.v0
        self.n_z0               = domain_object.n_z0
        self.n_app              = domain_object.n_app
        # -----
        self.call_level    = 0
        # -----
        self.E_totoal_prev = float("inf")
        self.Ea_prev       = float("inf")
        self.En_prev       = float("inf")

    def initialize_Rt_with_R0(self, R0: np.ndarray):
        # self.Rt = R0
        self.R0 = R0
        self.text_logger.set_R0(R0)

    def update_E_total_prev(self, E_total: float):
        self.E_totoal_prev = E_total

    def update_Ea_prev(self, Ea: float):
        self.Ea_prev = Ea

    def update_En_prev(self, En: float):
        self.En_prev = En

    def optimize(self,
            source_set    : SourcePointSurfaceSet,
            target_set    : TargetPointSurfaceSet,
            # ----
            delta_d       : np.ndarray,
        ):
        # ---------------------- normal vector -----------------------
        n_z = (self.R0 @ self.n_z0)
        v   = (self.R0 @ self.v0)
        # ---------------- solve least square problem ----------------
        rotvec_est, translation_est = self.visf_palm_Rt_ls.solve_Rt(
            source  = source_set.correspondence,
            target  = target_set.correspondence,
            delta_d = delta_d,
            v       = v,
            n_z     = n_z,
        )
        self.rotvec_est = rotvec_est
        # ----
        rotation_est = ExtendedRotation.from_rotvec(rotvec_est)
        R_rod        = rotation_est.as_rodrigues()
        R_skew       = rotation_est.as_skew()
        # --------- get optimal transformed source points (rodrigues) ---------
        aligned_source_set_skew = transform_source_set(source_set, R_skew, translation_est, delta_d, v)
        aligned_source_set_rod  = transform_source_set(source_set, R_rod, translation_est, delta_d, v)
        aligned_n_z             = (R_rod @ n_z)
        # aligned_v               = (R_rod @ v)
        # -------------- update current Rotation (Rt) ----------------
        # self.update_Rt(R_rod)
        # -----
        # errors_rodrigues        = self.error.compute(aligned_source_set_rod, target_set)
        alpha_En = self.En.compute_with_weight(aligned_source_set_rod, target_set, history_append=True)
        beta_Ea  = self.Ea.compute_with_weight(n_z=aligned_n_z, history_append=True)
        E_total  = (alpha_En + beta_Ea)
        # print(f"E_total = {E_total:.7f}")

        # -------------- update current Rotation (Rt) ----------------
        self.update_E_total_prev(E_total)
        self.update_En_prev(alpha_En)
        self.update_Ea_prev(beta_Ea)

        # ---------------------------- verbose -------------------------------
        self.text_logger.palm_opt_finished(rotation_est, translation_est)

        # ---------------------------- visualize -------------------------------
        # self.dipfo_visualizer.visualize_with_skew(
        #     source_set_skew = aligned_source_set_skew,
        #     source_set_rod  = aligned_source_set_rod,
        #     target_set      = target_set,
        #     n_z             = aligned_n_z,
        #     call_level      = self.call_level,
        #     title           = "[IPFO] R*,t*: skew & rodrigues",
        # )
        # ----
        self.isf_visualizer.visualize(
            source_set = aligned_source_set_rod,
            target_set = target_set,
            n_z        = aligned_n_z,
            call_level = self.call_level,
            title="[IPFO] R*t*: palm opt",
        )
        # ----------
        return R_rod, translation_est, aligned_n_z



