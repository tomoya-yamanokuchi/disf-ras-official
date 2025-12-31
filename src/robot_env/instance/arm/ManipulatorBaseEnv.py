import numpy as np
import mujoco
import time
from domain_object.builder import DomainObject
from print_color import print

class ManipulatorBaseEnv:
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
        self.ik_solver     = domain_object.ik_solver
        # ---
        self.hand_dof_ids  = np.array([self.model.joint(name).id for name in self.config.hand_joint_names])
        # ---
        self.initialized_with_keyframe = False

    def time(self):
        return self.data.time # [sec]

    def set_home_keyframe(self,
        key_name    : str               = "home",
        object_pos  : np.ndarray | None = None,
        object_quat : np.ndarray | None = None,
    ):
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, key_name)
        key_qpos = self.model.key_qpos.reshape(self.model.nkey, self.model.nq)
        key_ctrl = self.model.key_ctrl.reshape(self.model.nkey, self.model.nu)

        # ----------------------------------------------
        key_object_qpos   = key_qpos[key_id, -7:]
        # ----------------------------------------------
        key_object_pos  = key_object_qpos[:3]
        key_object_quat = key_object_qpos[3:7]
        # ----------------------------------------------

        if object_pos is not None:
            object_pos = np.array(object_pos, dtype=float)
            key_object_pos[:] = object_pos[:]

            print(f"object_pos is replaced with [{object_pos}]",
                  tag = "home keyframe (object_pos)", color="y", tag_color="y")
        else:
            print(f"Please give the object_pos information for table hight",
                tag = "home keyframe (object_pos)", color="m", tag_color="m")
            raise NotImplementedError()

        if object_quat is not None:
            object_quat = np.array(object_quat, dtype=float)
            key_object_quat[:] = object_quat[:]

            print(f"object_quat is replaced with [{object_quat}]",
                  tag = "home keyframe (object_quat)", color="y", tag_color="y")
        else:
            print(f"Please give the object_quat information for table hight",
                tag = "home keyframe (object_quat)", color="m", tag_color="m")
            raise NotImplementedError()

        # import ipdb; ipdb.set_trace()


    def reset(self, keyframe: int = None):
        # ----------------------------------------------
        self.arm_dof_ids      = np.array([self.model.joint(name).id    for name in self.config.arm_joint_names])
        self.arm_actuator_ids = np.array([self.model.actuator(name).id for name in self.config.arm_actuator_names])
        self.hand_dof_ids     = np.array([self.model.joint(name).id    for name in self.config.hand_joint_names])
        # ----------------------------------------------
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

    def get_mujoco_viewer_params(self, show_ui: bool = False):
        return self.mujoco_viewer.get_viewer_params(show_ui)

    def initialize_ik_solver(self):
        self.ik_solver.initialize()
        self.ik_solver.set_dof_ids(self.arm_dof_ids)
        self.ik_solver.set_actuator_ids(self.arm_actuator_ids)
        # set id of end effector to be controlled
        end_effector_site_id = self.model.site(self.config.end_effector_site_name).id
        self.ik_solver.set_site_id(end_effector_site_id)
        # set id of mocap site for generating targtet pose
        # Mocap body we will control with our mouse.

        # import ipdb; ipdb.set_trace()

        # mocap_site_id = self.model.body(self.config.mocap_site_name).mocapid[0]
        # self.ik_solver.set_mocap_id(mocap_site_id)


    def get_quat_arm(self):
        return self.data.qpos[self.arm_dof_ids]

    def solve_ik(self):
        self.ik_solver.solve()

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

    def get_object_joint_id(self):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.object_joint_name)

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
        assert qpos.shape == (len(self.hand_dof_ids),)
        self.data.ctrl[self.hand_dof_ids] = qpos

