import os
from domain_object.builder import DomainObject
from robot_env.instance.arm.ManipulatorGraspEnv import ManipulatorGraspEnv


class MujocoHandAloneEnv:
    def __init__(self, domain_object: DomainObject):
        self.geom = domain_object.geom
        self.viewer_wrapper                 = domain_object.viewer_wrapper
        self.env              = domain_object.env

    def execute(self):
        self.env.reset(keyframe=0)
        self.env.set_params()
        # -----------------------------------------
        self.viewer_wrapper.launch()
        self.viewer_wrapper.initialize_for_env()
        # 2) with ブロックで使う
        with self.viewer_wrapper as viewer:
            viewer.camera.set_overview()
            # # -----
            while True:
                self.env.start_step()
                # env.solve_ik()
                self.env.step()
                self.env.wait_step()
                viewer.sync()
