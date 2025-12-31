import numpy as np
from value_object import IPFOErrors
from service import ExtendedRotation
from domain_object.builder import DomainObject
from service import transform_source_set
from value_object import IPFOResult, PFOResult
from value_object import SourcePointSurfaceSet, TargetPointSurfaceSet
import time
import cma


class CMASF:
    def __init__(self, domain_object: DomainObject):
        # ---
        self.isf_visualizer       = domain_object.isf_visualizer
        # ---
        self.isf_loop_criteria    = domain_object.isf_loop_criteria
        self.text_logger          = domain_object.text_logger
        self.finger_indices       = domain_object.finger_indices
        self.error                = domain_object.error
        self.geom_error           = domain_object.geom_error
        self.com_error            = domain_object.com_error
        self.v0                   = domain_object.v0
        self.save_dir             = domain_object.save_dir
        self.verbose              = domain_object.verbose
        self.n_z0                 = domain_object.n_z0
        self.d0                   = domain_object.d0
        self.d_min                = domain_object.d_min
        self.d_max                = domain_object.d_max
        # ----
        self.initial_mean    = np.deg2rad(domain_object.config_cma.initial_mean_degree)
        self.initial_std_dev = domain_object.config_cma.initial_std_dev
        self.n_jobs          = domain_object.config_cma.n_jobs
        # ----
        self.contact_indices           = domain_object.contact_indices
        # ----
        self.call_level = 1


    def run(self, params: np.ndarray):
        # print(f"****** params = {params}")
        # ---------------------
        rotvec  = params[0:3]
        t       = params[3:6]
        delta_d = params[6:]
        # ---------------------
        R = ExtendedRotation.from_rotvec(rotvec).as_rodrigues()
        aligned_source_set = transform_source_set(self.source_set, R, t, delta_d, self.v)
        aligned_n_z        = (R @ self.n_z)
        err                = self.error.compute(aligned_source_set, self.target_set, aligned_n_z, history_append=False)
        # ---
        # print(f"err.total = {err.total}")
        # ---------------------
        # import ipdb; ipdb.set_trace()
        return err.total

    def optimize(self,
            source_set: SourcePointSurfaceSet,
            target_set: TargetPointSurfaceSet,
            R0        : np.ndarray,
        ) -> IPFOResult:
        # ---------------------------------------------------------
        d               = self.d0
        self.source_set = source_set
        self.target_set = target_set
        self.n_z        = (R0 @ self.n_z0)
        self.v          = (R0 @ self.v0)
        # ---------------------------------------------------------
        # 各パラメータごとの境界

        lower_bounds    = np.array([-np.pi, -np.pi, -np.pi, -0.1, -0.1, -0.1, 0.0])
        upper_bounds    = np.array([ np.pi,  np.pi,  np.pi,  0.1,  0.1,  0.1, self.d_max])
        initial_std_dev = np.zeros(7) + 1.0

        sigma0   = np.mean(upper_bounds - lower_bounds) * 0.1
        CMA_stds = (upper_bounds - lower_bounds) / np.max(upper_bounds - lower_bounds)

        # import ipdb; ipdb.set_trace()
        # ----
        assert np.all(lower_bounds < upper_bounds), "Error: lower_bounds must be smaller than upper_bounds!"
        # assert np.all(initial_std_dev > 0), "Error: CMA_stds must be positive!"
        # CMA-ESのオプション設定
        options = {
            'bounds': [lower_bounds, upper_bounds],  # 各パラメータの境界を設定
            'CMA_stds': CMA_stds,  # パラメータごとに異なる標準偏差
        }
        es = cma.CMAEvolutionStrategy(
            x0      = self.initial_mean,
            sigma0  = sigma0,
            options = options,
        )
        # --------------- optimize with time mesure ---------------
        start_time = time.time()
        es.optimize(
            objective_fct = self.run,
            n_jobs        = 0,
        )
        end_time     = time.time()
        elapsed_time = (end_time - start_time)
        # -----
        result_cma = es.result
        # ---------------------------------------------------------
        rotvec_opt  = result_cma.xbest[0:3]
        t_opt       = result_cma.xbest[3:6]
        delta_d_opt = result_cma.xbest[6:]
        # ----
        R_opt = ExtendedRotation.from_rotvec(rotvec_opt).as_rodrigues()
        aligned_source_set = transform_source_set(source_set, R_opt, t_opt, delta_d_opt, self.v)
        aligned_n_z        = (R_opt @ self.n_z)
        e_p_sum            = self.error.compute(aligned_source_set, target_set, aligned_n_z)
        # ----
        e_geom             = self.geom_error.compute(aligned_source_set, target_set)
        e_com              = self.com_error.compute(aligned_source_set, target_set)
        # ----
        self.text_logger.error_diff_in_ipfo(source_set, aligned_source_set)
        # ----------------- resut ------------------
        result = IPFOResult(
            R_sum                  = R_opt,
            t_sum                  = t_opt,
            delta_d_sum            = delta_d_opt,
            d                      = d,
            e_geom                 = e_geom,
            e_com                  = e_com,
            e_p_sum                = e_p_sum,
            aligned_source_set     = aligned_source_set,
            aligned_n_z            = aligned_n_z,
            elapsed_time           = elapsed_time,
        )
        return result
