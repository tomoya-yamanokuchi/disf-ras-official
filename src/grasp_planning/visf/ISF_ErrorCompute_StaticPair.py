import os
import numpy as np
from service import ExtendedRotation
from domain_object.builder import DomainObject
from value_object import ISFResult
from value_object import SourcePointSurfaceSet, TargetPointSurfaceSet



class ISF_ErrorCompute_StaticPair: # ISF
    def __init__(self, domain_object: DomainObject):
        self.ipfo_error_computation    = domain_object.ipfo_error_computation
        self.dipfo_visualizer          = domain_object.dipfo_visualizer
        # ---
        self.visualizer                = domain_object.visualizer
        self.icp_matcher               = domain_object.icp_matcher
        self.rigidTransMatBuilder      = domain_object.rigidTransMatBuilder
        self.loop_criteria             = domain_object.loop_criteria
        self.text_logger               = domain_object.text_logger
        self.surface_dataset_generator = domain_object.surface_dataset_generator
        self.d0                        = domain_object.d0
        # ----
        self.source = domain_object.source
        # ----
        self.gripper_euler_angle_rad   = domain_object.gripper_euler_angle_rad
        self.gripper_translation       = domain_object.gripper_translation
        # ----
        self.rotvec_rad_object         = domain_object.rotvec_rad_object
        self.v0                        = domain_object.v0
        self.n_z0                      = domain_object.n_z0
        self.model_name                = domain_object.model_name
        # ---
        self.contact_indices = domain_object.contact_indices
        # ----
        self.object_contact_surface = domain_object.object_contact_surface #  centered_ycb_contact_point_normal
        self.object_whole_surface   = domain_object.object_whole_surface   #  centered_ycb_point_normal
        # ---
        self.call_level_inner = 2
        self.call_level_outer = 3
        # ----
        self.Ea = domain_object.Ea
        self.En = domain_object.En
        self.Ep = domain_object.Ep


    def run(self, rotvec0: np.ndarray = None, return_all: bool = False) -> ISFResult:
        if rotvec0 is None:
            R0 = ExtendedRotation.from_euler(self.gripper_euler_angle_rad).as_rodrigues()
        else:
            # import ipdb; ipdb.set_trace()
            R0 = ExtendedRotation.from_rotvec(rotvec0).as_rodrigues()
        # ----------------------------
        t0 = self.gripper_translation
        # ----------------------------
        source_surface = source_unit_rigid_transformation_from_Rt(
            source_unit=self.source, R=R0, t=t0,
        )
        target_surface = self.object_contact_surface
        # ---------------------------
        es      = float("inf")
        es_prev = float("inf")
        Sf0     = source_surface.points
        eta     = 0.0
        count   = 0
        d       = self.d0
        # -----------
        R_sum_isf       = np.eye(3)
        t_sum_isf       = np.zeros(3)
        delta_d_sum_isf = 0.0
        # -----------
        es_hist = []
        # ================== iterative optimization ==================
        self.ipfo_error_computation.ipfo_palm_Rt_opt.initialize_Rt_with_R0(R0=R0)
        self.dipfo_visualizer.set_target_information(
            object_whole_surface = self.object_whole_surface,
            contact_indices      = self.contact_indices,
        )
        self.icp_matcher.set_target(target_surface)
        # -------
        icp_result = self.icp_matcher.find_correspondences(source_surface)
        source_set = SourcePointSurfaceSet(correspondence=icp_result.source, surface=source_surface)
        target_set = TargetPointSurfaceSet(
            correspondence  = icp_result.target,
            contact_surface = target_surface,
            whole_surface   = self.object_whole_surface
        )
        # -------
        self.dipfo_visualizer.visualize(
            source_set = source_set,
            target_set = target_set,
            n_z        = (R0 @ self.n_z0),
            call_level = self.call_level_outer,
            title      = f"[DISF] init",
        )


        # ==============================================================================
        result = self.ipfo_error_computation.compute(source_set, target_set, d)
        # ==============================================================================


