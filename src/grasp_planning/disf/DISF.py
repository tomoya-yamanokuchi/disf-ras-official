import numpy as np
from value_object import IPFOErrors
from domain_object.builder import DomainObject
from service import transform_source_set
from service import update_R, update_t, update_delta_d
from value_object import IPFOResult, PFOResult
from service import ExtendedRotation
from value_object import SourcePointSurfaceSet, TargetPointSurfaceSet
import time

class DISF: # PFO
    def __init__(self, domain_object: DomainObject):
        self.disf_palm_R_opt     = domain_object.disf_palm_R_opt
        self.disf_trans_centroid = domain_object.disf_trans_centroid
        self.disf_finger_opt     = domain_object.disf_finger_opt
        # ---
        self.isf_visualizer      = domain_object.isf_visualizer
        self.isf_loop_criteria   = domain_object.isf_loop_criteria
        self.text_logger         = domain_object.text_logger
        self.finger_indices      = domain_object.finger_indices
        self.error               = domain_object.error
        self.geom_error           = domain_object.geom_error
        self.com_error           = domain_object.com_error
        self.v0                  = domain_object.v0
        self.save_dir            = domain_object.save_dir
        self.verbose             = domain_object.verbose
        self.n_z0                = domain_object.n_z0
        self.d0                  = domain_object.d0
        # ----
        self.contact_indices           = domain_object.contact_indices
        # ----
        self.call_level = 1


    def optimize(self,
            source_set: SourcePointSurfaceSet,
            target_set: TargetPointSurfaceSet,
            R0        : np.ndarray,
        ) -> IPFOResult:
        # ---------------------------------------------------------
        d = self.d0
        self.disf_palm_R_opt.initialize_Rt_with_R0(R0=R0)
        # ---------------------------------------------------------
        source_set0 = source_set
        Rt0         = self.disf_palm_R_opt.Rt
        n_z0        = (Rt0 @ self.n_z0)
        v0          = (Rt0 @ self.v0)
        # ---------------------------------------------------------
        e_p         = IPFOErrors(float('inf'), None, None, None)
        e_t         = IPFOErrors(0.0, None, None, None)
        R_sum       = np.eye(3)
        t_sum       = np.zeros(3)
        delta_d_sum = 0.0
        # -------------------
        n_z = n_z0
        v   = v0
        # ---
        self.error.compute(source_set, target_set, n_z, history_append=True, history_append_sub=True) # initial error
        self.isf_loop_criteria.reset_count()
        # ----
        start_time = time.time()
        while self.isf_loop_criteria.evaluate(e_p, e_t):
            e_p = self.error.compute(source_set, target_set, n_z, history_append=False)
            # ----
            source_set, R, n_z, v = self.disf_palm_R_opt.optimize(source_set, target_set, n_z, v)
            source_set, t         = self.disf_trans_centroid.optimize(source_set, target_set, n_z, v)
            source_set, delta_d   = self.disf_finger_opt.optimize(source_set, target_set, d, n_z, v)
            # ---
            d = (d + delta_d)
            # ---
            e_t = self.error.compute(source_set, target_set, n_z, history_append=True)
            # import ipdb; ipdb.set_trace()
            # ----
            R_sum       = update_R(R=R, Rt=R_sum)
            t_sum       = update_t(t=t, R=R, t_t=t_sum)
            delta_d_sum = update_delta_d(delta_d, delta_d_sum)
            # --------------
            self.isf_loop_criteria.add_count()
        # ----
        end_time     = time.time()
        elapsed_time = (end_time - start_time)
        # ---------------------------------------------------------
        aligned_source_set = transform_source_set(source_set0, R_sum, t_sum, delta_d_sum, v0)
        aligned_n_z        = (R_sum @ n_z0)
        e_p_sum            = self.error.compute(aligned_source_set, target_set, n_z)
        # ----
        e_geom             = self.geom_error.compute(aligned_source_set, target_set)
        e_com              = self.com_error.compute(aligned_source_set, target_set)
        # ----
        self.text_logger.error_diff_in_ipfo(source_set, aligned_source_set)
        # ----------------- resut ------------------
        result = IPFOResult(
            R_sum                  = R_sum,
            t_sum                  = t_sum,
            delta_d_sum            = delta_d_sum,
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
        # -------------------------------------------
        self.disf_palm_R_opt.Ea.save_history()
        self.disf_palm_R_opt.En.save_history()
        self.disf_finger_opt.Ep.save_history()
        self.error.save_history()
        # ----
        return result
