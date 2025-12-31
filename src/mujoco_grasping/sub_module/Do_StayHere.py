import os
import numpy as np
from print_color import print
from domain_object.builder import DomainObject
from robot_env.instance.arm.ManipulatorGraspEnv import ManipulatorGraspEnv
from robot_env.utils import MyViewerWrapper
from .TimeWatch import TimeWatch

class Do_StayHere:
    def __init__(self, domain_object: DomainObject):
        self.env : ManipulatorGraspEnv = domain_object.env
        self.time_watch                = TimeWatch(domain_object)
        self.grasp_evaluator           = domain_object.grasp_evaluator

    def execute(self,
            viewer   : MyViewerWrapper,
            stay_step: int):
        # ---------------------------------------
        # if self.grasp_evaluator.safety_violation:
        #     return
        # ---------------------------------------
        # while True:
        for i in range(stay_step):
            self.env.start_step()
            self.env.step()
            self.env.wait_step()
            viewer.sync()
        # ----
        print(f"stay finish!",
                tag = "Grasp", color="c", tag_color="c")
