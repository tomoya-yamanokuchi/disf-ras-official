import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R


class LevenbergMarquardt:
    def __init__(self, model, data, config):
        self.model           = model
        self.data            = data
        # ---
        self.step_size       = config.step_size
        self.tol             = config.tol
        self.damping         = config.damping

    def set_init_qpos(self):
        self.init_qpos = self.data.qpos.copy()

    def get_end_effector_id(self, body_name: str):
        return self.model.body(body_name).id # End-effector we wish to control.

    def set_end_effector_id(self, end_effector_id: int):
        self.end_effector_id = end_effector_id

    def get_end_effector_geom_id(self, geom_name: str):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)

    def set_end_effector_geom_id(self, end_effector_geom_id: int):
        self.end_effector_geom_id = end_effector_geom_id

    def get_end_effector_body_id(self, end_effector_body_name: str):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, end_effector_body_name)

    def set_end_effector_body_id(self, end_effector_body_id: str):
        self.end_effector_body_id = end_effector_body_id

    def initialize_jacobian(self):
        self.jacp = np.zeros((3, self.model.nv)) # translation jacobian
        self.jacr = np.zeros((3, self.model.nv)) # rotational jacobian

    def clip_joint_limits(self):
        q = self.data.qpos
        for i in range(len(q)):
            q[i] = max(self.model.jnt_range[i][0], min(q[i], self.model.jnt_range[i][1]))

    def set_target_position(self, target_pos):
        self.target_pos = target_pos

    def set_target_quaternion(self, target_quat):
        self.target_quat = target_quat

    def update_pose_error(self):
        current_pos  = self.data.xpos[self.end_effector_body_id]
        current_quat = self.data.xquat[self.end_effector_body_id]
        # import ipdb ; ipdb.set_trace()
        pos_error    = (self.target_pos - current_pos)
        quat_error   = R.from_quat(self.target_quat).inv() * R.from_quat(current_quat)
        quat_error   = quat_error.as_rotvec()
        pose_error   = np.concatenate([pos_error, quat_error]) # 6dim
        self.error   = pose_error

    # def update_rotational_error(self):
    #     pos_error    = np.zeros(3)
    #     current_quat = self.data.xquat[self.end_effector_body_id]
    #     quat_error   = R.from_quat(self.target_quat).inv() * R.from_quat(current_quat)
    #     quat_error   = quat_error.as_rotvec()
    #     pose_error   = np.concatenate([pos_error, quat_error]) # 6dim
    #     self.error   = pose_error

    def get_error_norm(self):
        return np.linalg.norm(self.error)

    def check_tolerance(self):
        """
            - error is equal to tolerance or less -> True
            - otherwise                           -> False
        """
        return self.get_error_norm() <= self.tol

    def update_jacobian(self):
        mujoco.mj_jac(self.model, self.data, self.jacp, self.jacr, self.goal, self.end_effector_id)

    def update_jacobian_body(self):
        mujoco.mj_jacBody(self.model, self.data, self.jacp, self.jacr, self.end_effector_body_id)

    def update_target_pose(self):
        jacobian = np.vstack([self.jacp, self.jacr])
        identity = np.eye(jacobian.shape[1])
        jt_j     = (jacobian.T @ jacobian)
        product  = jt_j + (self.damping * identity)
        j_inv    = np.linalg.pinv(product) @ jacobian.T
        delta_q  = j_inv @ self.error
        # ---
        # q = self.init_qpos.copy()
        q = self.data.qpos.copy()
        q += self.step_size * delta_q
        # ---
        self.clip_joint_limits()
        self.data.ctrl = q


