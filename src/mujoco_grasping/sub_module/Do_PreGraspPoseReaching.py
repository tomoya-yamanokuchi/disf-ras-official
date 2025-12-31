import os
import numpy as np
from print_color import print
from domain_object.builder import DomainObject
from robot_env.instance.arm.ManipulatorGraspEnv import ManipulatorGraspEnv
from robot_env.utils import MyViewerWrapper
from .TimeWatch import TimeWatch


class Do_PreGraspPoseReaching:
    def __init__(self, domain_object: DomainObject):
        self.env : ManipulatorGraspEnv = domain_object.env
        self.time_watch        = TimeWatch(domain_object)


    def execute(self,
            viewer        : MyViewerWrapper,
            pre_grasp_pos : np.ndarray,
            pre_grasp_quat: np.ndarray,
        ):
        self.env.initialize_ik_solver()
        self.env.ik_solver.update_target_pose(
            target_pos  = pre_grasp_pos,
            target_quat = pre_grasp_quat,
        )
        self.env.ik_solver.set_reaching_threshold_pre_grasp()
        self.env.ik_solver.reset_ik_reach_flag()
        # ---
        self.time_watch.start()
        # ---------
        while not self.env.ik_solver.reached:
            # ----
            self.env.start_step()
            self.env.solve_ik()
            self.env.step()
            self.env.wait_step()
            # ----
            viewer.sync()
            # -----
            if not self.time_watch.check_continue():
                break
        # ---------
        if self.env.ik_solver.reached:
            print(f"reached at pre-grasp pose!",
                  tag = "Grasp", color="c", tag_color="c")
        # ---------
        # import ipdb; ipdb.set_trace()
