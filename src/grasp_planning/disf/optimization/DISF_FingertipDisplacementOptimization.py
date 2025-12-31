import numpy as np
from domain_object.builder import DomainObject
from value_object import SourcePointSurfaceSet, TargetPointSurfaceSet
from service import transform_source_set


class DISF_FingertipDisplacementOptimization:
    def __init__(self, domain_object: DomainObject):
        self.disf_palm_R_opt      = domain_object.disf_palm_R_opt
        self.finger_ls_Ep         = domain_object.disf_finger_ls_Ep
        # -----
        self.isf_visualizer      = domain_object.isf_visualizer
        self.finger_indices       = domain_object.finger_indices
        self.Ep                   = domain_object.Ep
        self.verbose              = domain_object.verbose
        self.text_logger          = domain_object.text_logger
        self.g                    = domain_object.v0
        self.n_z0                 = domain_object.n_z0
        # ---
        self.call_level           = 0


    def optimize(self,
            source_set : SourcePointSurfaceSet,
            target_set : TargetPointSurfaceSet,
            # ----
            d          : float,
            # ----
            n_z        : np.ndarray,
            v          : np.ndarray,
        ):
        # -----------------------------------------------------------
        R_I = np.eye(3)
        t_0 = np.zeros(3)
        # ---------------- solve least square problem ----------------
        delta_d_est = self.finger_ls_Ep.solve_delta_d(
            source = source_set.correspondence,
            target = target_set.correspondence,
            v      = v,
            d      = d,
        )
        # import ipdb; ipdb.set_trace()
        # -------------- compute transformation --------------
        aligned_source_set = transform_source_set(source_set, R_I, t_0, delta_d_est, v)
        # ----------- error computation -----------
        Ep = self.Ep.compute(aligned_source_set, target_set, history_append=True)
        # --------------------- textlog -----------------------
        self.text_logger.finger_opt_finished(delta_d_est)
        # ------------------ visualization ---------------------
        self.isf_visualizer.visualize(
            source_set = aligned_source_set,
            target_set = target_set,
            n_z        = n_z,
            call_level = self.call_level,
            title="[DIPFO] delta_d*: fingertip opt",
        )
        # ------------------------------------------------------
        return aligned_source_set, delta_d_est


