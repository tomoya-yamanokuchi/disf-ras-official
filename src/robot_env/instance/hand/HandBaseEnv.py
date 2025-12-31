import numpy as np
import mujoco
import time
from domain_object.builder import DomainObject


class HandBaseEnv:
    def __init__(self, domain_object: DomainObject):
        self.env_name      = "dexterous_hand"
        self.config        = domain_object.config_env
        self.dt            = self.config.option.timestep
        self.model         = domain_object.model
        self.data          = domain_object.data
        self.body          = domain_object.body
        self.geom          = domain_object.geom
        self.site          = domain_object.site
        # ---
        # self.ik_solver     = domain_object.ik_solver
        # ---
        self.hand_dof_ids  = np.array([self.model.joint(name).id for name in self.config.hand_joint_names])
        # ---
        self.initialized_with_keyframe = False

    def reset(self, keyframe: int = None):
        if (keyframe is None) or (self.model.nkey == 0):
            print("keyframe is None")
            mujoco.mj_resetData(self.model, self.data)
        else:
            print(f"keyframe = {keyframe}")
            mujoco.mj_resetDataKeyframe(self.model, self.data, keyframe)
            self.initialized_with_keyframe = True
        self.forward()

    def initialize_mujoco_viewer(self, viewer):
        self.mujoco_viewer.initialize(viewer)

    def get_mujoco_viewer_params(self, show_ui: bool = None):
        return self.mujoco_viewer.get_viewer_params(show_ui)

    def forward(self):
        mujoco.mj_forward(self.model, self.data)

    def start_step(self):
        self.step_start_time = time.time()

    def step(self):
        mujoco.mj_step(self.model, self.data)

    def wait_step(self):
        time_until_next_step = self.dt - (time.time() - self.step_start_time)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

    def get_robot_qpos(self):
        qpos_address = self.model.jnt_qposadr[self.get_object_joint_id()]
        qpos_value   = self.data.qpos[:qpos_address]
        return qpos_value

    def set_qpos(self, qpos: np.ndarray):
        self.data.qpos[:] = qpos

    def set_qpos_hand(self, qpos: np.ndarray):
        assert qpos.shape == (len(self.hand_dof_ids),)
        self.data.qpos[self.hand_dof_ids] = qpos

    def set_ctrl_hand(self, qpos: np.ndarray):
        # import ipdb ; ipdb.set_trace()
        # assert qpos.shape == (len(self.hand_dof_ids),)
        self.data.ctrl[:] = qpos
