import numpy as np
from service import ExtendedRotation
from value_object import IPFOParams, PointNormalUnitPairs
from domain_object.builder import DomainObject
from value_object import PFOTransformation
from value_object import PointNormalIndexUnitPairs
from value_object import SourcePointSurfaceSet, TargetPointSurfaceSet
from service import transform_fingertip, transform_source_set


class IPFO_FingertipDisplacementOptimization:
    def __init__(self, domain_object: DomainObject):
        self.ipfo_palm_Rt_opt    = domain_object.visf_palm_Rt_opt
        self.ipfo_finger_ls      = domain_object.visf_finger_ls
        # ------------------------------------------------------------
        self.isf_visualizer    = domain_object.isf_visualizer
        # -----
        self.finger_indices      = domain_object.finger_indices
        # self.error               = domain_object.error
        self.Ep                 = domain_object.Ep
        self.verbose             = domain_object.verbose
        self.text_logger         = domain_object.text_logger
        self.v0                  = domain_object.v0
        self.n_z0                = domain_object.n_z0
        # ---
        self.call_level = 0


    def optimize(self,
            source_set : SourcePointSurfaceSet,
            target_set : TargetPointSurfaceSet,
            # ----
            R          : np.ndarray,
            t          : np.ndarray,
            d          : float,
        ):
        # ---------------------- normal vector -----------------------
        R0  = self.ipfo_palm_Rt_opt.R0
        n_z = (R0 @ self.n_z0)
        v   = (R0 @ self.v0)
        # ---------------- solve least square problem ----------------
        delta_d_est = self.ipfo_finger_ls.solve_delta_d(
            source = source_set.correspondence,
            target = target_set.correspondence,
            R      = R,
            t      = t,
            v      = v,
            d      = d,
        )
        # import ipdb; ipdb.set_trace()
        # -------------- compute transformation --------------
        aligned_source_set = transform_source_set(source_set, R, t, delta_d_est, v)
        aligned_n_z        = (R @ n_z)
        # ----------- error computation -----------
        # errors = self.error.compute(aligned_source_set, target_set)
        Ep = self.Ep.compute(aligned_source_set, target_set, history_append=True)
        # --------------------- textlog -----------------------
        self.text_logger.finger_opt_finished(delta_d_est)
        # ------------------ visualization ---------------------
        self.isf_visualizer.visualize(
            source_set = aligned_source_set,
            target_set = target_set,
            n_z        = aligned_n_z,
            call_level = self.call_level,
            title="[DIPFO] delta_d*: fingertip opt",
        )
        # ------------------------------------------------------
        return delta_d_est


