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


class IPFO_ErrorCompute:
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
        num_rotation   = 100
        rotation_array = np.linspace(start=0, stop=np.pi, num=num_rotation)
        np.save(
            file = os.path.join(self.error.results_save_dir, "x.npy"),
            arr  = rotation_array,
        )
        rot_axis       = np.array([0, 0, 1])
        # -----
        for i in range(num_rotation):
            # --------------
            theta  = rotation_array[i]
            rotvec = (rot_axis * theta)
            R      = ExtendedRotation.from_rotvec(rotvec).as_rodrigues()
            # --------------
            aligned_source_set = transform_source_set(source_set, R, t, delta_d, v)
            # self.dipfo_visualizer.visualize(aligned_source_set, target_set, (R @ n_z), self.call_level, title=f"[IPFO] i={i} : theta ={theta}")
            # -------------
            e_t = self.error.compute(aligned_source_set, target_set, n_z, history_append=True, history_append_sub=True)
        # ----------------- resut ------------------
        self.error.save_history()
        # import ipdb; ipdb.set_trace()
        # return result

        # self.dipfo_visualizer.visualize(
        #     source_set = aligned_source_set,
        #     target_set = target_set,
        #     n_z        = (R @ n_z),
        #     call_level = self.call_level,
        #     title      = "[IPFO] opt"
        # )
