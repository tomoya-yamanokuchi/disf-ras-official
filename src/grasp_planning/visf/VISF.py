import numpy as np
from value_object import IPFOErrors
from domain_object.builder import DomainObject
from service import transform_source_set
from value_object import IPFOResult, PFOResult
from value_object import SourcePointSurfaceSet, TargetPointSurfaceSet
from copy import deepcopy
import time


class VISF:
    def __init__(self, domain_object: DomainObject):
        self.visf_palm_Rt_opt     = domain_object.visf_palm_Rt_opt
        self.visf_finger_opt      = domain_object.visf_finger_opt
        # ---
        self.isf_visualizer       = domain_object.isf_visualizer
        self.isf_loop_criteria    = domain_object.isf_loop_criteria
        self.text_logger          = domain_object.text_logger
        self.finger_indices       = domain_object.finger_indices
        self.geom_error           = domain_object.geom_error
        self.com_error            = domain_object.com_error
        self.error                = domain_object.error
        self.v0                   = domain_object.v0
        self.save_dir             = domain_object.save_dir
        self.verbose              = domain_object.verbose
        self.n_z0                 = domain_object.n_z0
        self.d0                   = domain_object.d0
        # ---
        self.call_level = 1


    def optimize(self,
            source_set: SourcePointSurfaceSet,
            target_set: TargetPointSurfaceSet,
            R0        : np.ndarray,
        ) -> IPFOResult:
        # ---------------------------------------------------------
        d = self.d0
        self.visf_palm_Rt_opt.initialize_Rt_with_R0(R0=R0)
        # ---------------------------------------------------------
        source_set0 = source_set
        Rt0         = self.visf_palm_Rt_opt.R0
        n_z0        = (Rt0 @ self.n_z0)
        v0          = (Rt0 @ self.v0)
        n_z         = n_z0
        v           = v0
        delta_d     = np.zeros(1)
        # ---------------------------------------------------------
        self.isf_visualizer.visualize(source_set, target_set, n_z0, self.call_level, title="[IPFO] init")
        # ---------------------------------------------------------
        e_p         = IPFOErrors(float('inf'), None, None, None)
        e_t         = IPFOErrors(0.0, None, None, None)
        # -------------------
        self.isf_loop_criteria.reset_count()
        # ----
        start_time = time.time()
        while self.isf_loop_criteria.evaluate(e_p, e_t):
            e_p = deepcopy(e_t)
            # --------------
            R, t, n_z = self.visf_palm_Rt_opt.optimize(source_set, target_set, delta_d)
            delta_d   = self.visf_finger_opt.optimize(source_set, target_set, R, t, d)
            # --------------
            aligned_source_set = transform_source_set(source_set, R, t, delta_d, v)
            e_t                = self.error.compute(aligned_source_set, target_set, n_z, history_append=True)
            # --------------
            self.isf_loop_criteria.add_count()
        # ----
        end_time     = time.time()
        elapsed_time = (end_time - start_time)
        # ---------------------------------------------------------
        aligned_source_set = transform_source_set(source_set0, R, t, delta_d, v)
        aligned_n_z        = (R @ n_z)
        e_p_sum            = self.error.compute(aligned_source_set, target_set, n_z)
        # ----
        e_geom             = self.geom_error.compute(aligned_source_set, target_set)
        e_com              = self.com_error.compute(aligned_source_set, target_set)
        # ----
        self.text_logger.error_diff_in_ipfo(source_set, aligned_source_set)
        # ----------------- resut ------------------
        result = IPFOResult(
            R_sum                  = R,
            t_sum                  = t,
            delta_d_sum            = delta_d,
            d                      = d,
            e_geom                 = e_geom,
            e_com                  = e_com,
            e_p_sum                = e_p_sum,
            aligned_source_set     = aligned_source_set,
            aligned_n_z            = aligned_n_z,
            elapsed_time           = elapsed_time,
        )
        # ---------------- verbose ------------------
        self.text_logger.dipfo_finished(result)
        # -------------- visualization --------------
        self.isf_visualizer.visualize(
            source_set = aligned_source_set,
            target_set = target_set,
            n_z        = aligned_n_z,
            call_level = self.call_level,
            title      = "[IPFO] opt"
        )
        # -------------------------------------------
        self.visf_palm_Rt_opt.Ea.save_history()
        self.visf_palm_Rt_opt.En.save_history()
        self.visf_finger_opt.Ep.save_history()
        self.error.save_history()

        return result
