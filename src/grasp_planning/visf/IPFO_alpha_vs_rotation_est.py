import numpy as np
from value_object import IPFOParams
from value_object import IPFOErrors
from domain_object.builder import DomainObject
from service import transform_fingertip, transform_source_set
from service import update_R, update_t, update_delta_d
from value_object import IPFOResult, PFOResult
from scipy.spatial.transform import Rotation
from service import ExtendedRotation
from value_object import SourcePointSurfaceSet, TargetPointSurfaceSet
from copy import deepcopy

import os

class IPFO_alpha_vs_rotation_est:
    def __init__(self, domain_object: DomainObject):
        self.ipfo_palm_Rt_opt     = domain_object.ipfo_palm_Rt_opt
        self.ipfo_finger_opt      = domain_object.ipfo_finger_opt
        # ---
        self.dipfo_visualizer     = domain_object.dipfo_visualizer
        # ---
        self.ipfo_loop_criteria   = domain_object.ipfo_loop_criteria
        self.text_logger          = domain_object.text_logger
        self.visualizer           = domain_object.visualizer
        self.finger_indices       = domain_object.finger_indices
        self.error                = domain_object.error
        self.v0                   = domain_object.v0
        self.save_dir             = domain_object.save_dir
        self.verbose              = domain_object.verbose
        self.n_z0                 = domain_object.n_z0
        # ---
        self.call_level = 1


    def compute(self,
            source_set: SourcePointSurfaceSet,
            target_set: TargetPointSurfaceSet,
            d         : float,
        ) -> IPFOResult:
        # ---------------------------------------------------------
        source_set0 = source_set
        Rt0         = self.ipfo_palm_Rt_opt.R0
        n_z0        = (Rt0 @ self.n_z0)
        v0          = (Rt0 @ self.v0)
        # ---------------------------------------------------------
        self.dipfo_visualizer.visualize(source_set, target_set, n_z0, self.call_level, title="[IPFO] init")
        # -------------------
        n_z = n_z0
        v   = v0
        # -------------------
        delta_d = np.zeros(1)
        t       = np.zeros(3)
        # -----
        num_alpha    = 100
        alpha_array  = np.linspace(start=0, stop=1.0, num=num_alpha)
        # -----
        specific_x_values = [0, 0.4, 0.8]
        indices = [np.argmin(np.abs(alpha_array - val)) for val in specific_x_values]
        indices = [int(val) for val in indices]
        # -----
        np.save(
            file = os.path.join(self.error.results_save_dir, "alpha_array.npy"),
            arr  = alpha_array,
        )
        # -----
        rotvec_est_array   = []
        # -----
        for i in range(num_alpha):
            # --------------
            self.ipfo_palm_Rt_opt.ipfo_palm_Rt_ls.ls_EpEn.En_params.alpha = alpha_array[i]
            # --------------
            R, t       = self.ipfo_palm_Rt_opt.optimize(source_set, target_set, delta_d)
            rotvec_est = self.ipfo_palm_Rt_opt.rotvec_est
            rotvec_est_array.append(rotvec_est)
            # print("rotvec_est = ", rotvec_est)

            # ---------
            # import ipdb; ipdb.set_trace()
            # import ipdb; ipdb.set_trace()
            # print(f"i={i}", indices)
            if i in indices:
                print(f"i={i} : alpha={alpha_array[i]}")
                # import ipdb; ipdb.set_trace()
                aligned_source_set = transform_source_set(source_set, R, t, delta_d, v)

                # import ipdb; ipdb.set_trace()q
                self.dipfo_visualizer.visualize(
                    source_set = aligned_source_set,
                    target_set = target_set,
                    n_z        = None, # (R @ n_z),
                    call_level = self.call_level,
                    title      = "",
                    filename   = f"point_cloud_result_alpha={alpha_array[i]:0e}.pdf",
                )
                # -------------

        # ----------------- resut ------------------
        # import ipdb; ipdb.set_trace()
        np.save(
            file = os.path.join(self.error.results_save_dir, "rotvec_est.npy"),
            arr  = np.vstack(rotvec_est_array),
        )
