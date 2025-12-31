import os
import numpy as np
from print_color import print
from domain_object.builder import DomainObject
from robot_env.instance.arm.ManipulatorGraspEnv import ManipulatorGraspEnv
from robot_env.utils import MyViewerWrapper
from .TimeWatch import TimeWatch

class Do_FingerReach:
    def __init__(self, domain_object: DomainObject):
        self.env : ManipulatorGraspEnv = domain_object.env
        self.time_watch                = TimeWatch(domain_object)
        self.grasp_evaluator           = domain_object.grasp_evaluator

    def execute(self,
            viewer     : MyViewerWrapper,
            qpos_finger: np.ndarray,
        ):
        # ---------------------------------------
        if self.grasp_evaluator.safety_violation:
            return
        # ---------------------------------------
        self.env.set_ctrl_finger(qpos_finger)
        self.env.reset_finger_reach_flag()
        # ---
        self.time_watch.start()
        # ---
        while not self.env._finger_grasped:
            self.env.start_step()
            # ----
            # self.env.check_finger_stop()
            # ----
            self.env.step()
            self.env.wait_step()
            viewer.sync()
            # -----
            if not self.time_watch.check_continue():
                break

        # import ipdb; ipdb.set_trace()
        # ----
        print(f"grasp finish!",
                tag = "Grasp", color="c", tag_color="c")
