import mujoco
import mujoco.viewer
import numpy as np
import time
from service import ExtendedRotation


class DampedLeastSquares:
    def __init__(self,
            model,
            data,
            config_env,
            config_ik_solver
        ):
        self.model          = model
        self.data           = data
        self.config_env     = config_env
        # ---
        self.integration_dt = config_ik_solver.integration_dt
        self.damping        = config_ik_solver.damping
        self.max_angvel     = config_ik_solver.max_angvel
        # ---
        self.reaching_threshold = config_ik_solver.reaching_threshold
        # ---
        if config_ik_solver.gravity_compensation:
            self.gravity_compensation_setting()
        # ---
        self.target_pos  = None
        self.target_quat = None
        # ---
        self.reached     = False


    def set_reaching_threshold_pre_grasp(self):
        self.pos_threshold = self.reaching_threshold.pre_grasp.position
        self.ori_threshold = self.reaching_threshold.pre_grasp.orientation

    def set_reaching_threshold_optimal_grasp(self):
        self.pos_threshold = self.reaching_threshold.optimal_grasp.position
        self.ori_threshold = self.reaching_threshold.optimal_grasp.orientation

    def set_reaching_threshold_lift_up(self):
        self.pos_threshold = self.reaching_threshold.lift_up.position
        self.ori_threshold = self.reaching_threshold.lift_up.orientation

    def gravity_compensation_setting(self):
        # Name of bodies we wish to apply gravity compensation to.
        body_ids = [self.model.body(name).id for name in self.config_env.arm_body_names]
        self.model.body_gravcomp[body_ids] = 1.0

    def set_dof_ids(self, dof_ids: list):
        self.dof_ids = dof_ids

    def set_actuator_ids(self, actuator_ids: list):
        self.actuator_ids = actuator_ids

    def set_site_id(self, id: int):
        # End-effector site we wish to control
        self.site_id = id

    def set_mocap_id(self, id: int):
        # Mocap body we will control with our mouse.
        self.mocap_id = id

    def initialize(self):
        # Pre-allocate numpy arrays.
        self.jac               = np.zeros((6, self.model.nv))
        self.diag              = self.damping * np.eye(6)
        self.error             = np.zeros(6)
        self.error_pos         = self.error[:3]
        self.error_ori         = self.error[3:]
        self.current_quat      = np.zeros(4)
        self.current_quat_conj = np.zeros(4)
        self.error_quat        = np.zeros(4)

    def get_robot_joint_range(self):
        return self.model.jnt_range[self.dof_ids].T

    def update_target_pose(self,
            target_pos: np.ndarray,
            target_quat: np.ndarray
        ):
        self.target_pos  = target_pos
        self.target_quat = target_quat
        # import ipdb; ipdb.set_trace()

    def update_target_pose_with_mocap(self):
        self.target_pos  = self.data.mocap_pos[self.mocap_id]
        self.target_quat = self.data.mocap_quat[self.mocap_id]

    def update_current_pose(self):
        self.current_pos  = self.data.site(self.site_id).xpos
        self.current_xmat = self.data.site(self.site_id).xmat # rotation info
        # print("current rotvec =", np.rad2deg(ExtendedRotation.from_matrix(self.current_xmat.reshape(3, 3)).as_rotvec()))

    def reset_ik_reach_flag(self):
        self.reached = False

    def solve(self):
        assert self.target_pos is not None
        assert self.target_quat is not None
        '''
        target_pos and target_quat should be updaed by user from usecase
        '''

        self.update_current_pose()

        # Position error.
        self.error_pos[:] = (self.target_pos - self.current_pos)

        # Orientation error.
        mujoco.mju_mat2Quat(self.current_quat, self.current_xmat)
        mujoco.mju_negQuat(self.current_quat_conj, self.current_quat) # Conjugate quaternion (opposite same rotation)
        mujoco.mju_mulQuat(self.error_quat, self.target_quat, self.current_quat_conj)
        mujoco.mju_quat2Vel(self.error_ori, self.error_quat, 1.0)

        # Get the Jacobian with respect to the end-effector site.
        mujoco.mj_jacSite(self.model, self.data, self.jac[:3], self.jac[3:], self.site_id)

        # Solve system of equations: J @ dq = error.
        dq = self.jac.T @ np.linalg.solve(self.jac @ self.jac.T + self.diag, self.error)

        # Scale down joint velocities if they exceed maximum.
        if self.max_angvel > 0:
            dq_abs_max = np.abs(dq).max()
            if dq_abs_max > self.max_angvel:
                dq *= self.max_angvel / dq_abs_max

        # Integrate joint velocities to obtain joint positions.
        q = self.data.qpos.copy()
        mujoco.mj_integratePos(self.model, q, dq, self.integration_dt)

        # Set the control signal.
        np.clip(q[self.dof_ids], *self.get_robot_joint_range(), out=q[self.dof_ids])
        self.data.ctrl[self.actuator_ids] = q[self.dof_ids]

        # print("self.data.ctrl = ", self.data.ctrl)

        # import ipdb; ipdb.set_trace()

        # ========= evalute if the arm is reached to target or not =========
        pos_error_norm      = np.linalg.norm(self.error_pos)
        ori_error_norm      = np.linalg.norm(self.error_ori)
        # print(f"pos_error_norm, ori_error_norm = [{pos_error_norm:.5f}, {ori_error_norm:.5f}]")
        # ---
        reached_position    = (pos_error_norm < self.pos_threshold)
        reached_orientation = (ori_error_norm < self.ori_threshold)
        # ---
        if reached_position and reached_orientation:
            self.reached = True #
            # print("self.reached = True #")
        # ===================================================================


    def solve_with_mocap(self):
        self.update_target_pose()
        self.update_current_pose()

        # Position error.
        self.error_pos[:] = (self.target_pos - self.current_pos)

        # Orientation error.
        mujoco.mju_mat2Quat(self.current_quat, self.current_xmat)
        mujoco.mju_negQuat(self.current_quat_conj, self.current_quat) # Conjugate quaternion (opposite same rotation)
        mujoco.mju_mulQuat(self.error_quat, self.target_quat, self.current_quat_conj)
        mujoco.mju_quat2Vel(self.error_ori, self.error_quat, 1.0)

        # Get the Jacobian with respect to the end-effector site.
        mujoco.mj_jacSite(self.model, self.data, self.jac[:3], self.jac[3:], self.site_id)

        # Solve system of equations: J @ dq = error.
        dq = self.jac.T @ np.linalg.solve(self.jac @ self.jac.T + self.diag, self.error)

        # Scale down joint velocities if they exceed maximum.
        if self.max_angvel > 0:
            dq_abs_max = np.abs(dq).max()
            if dq_abs_max > self.max_angvel:
                dq *= self.max_angvel / dq_abs_max

        # Integrate joint velocities to obtain joint positions.
        q = self.data.qpos.copy()
        mujoco.mj_integratePos(self.model, q, dq, self.integration_dt)

        # Set the control signal.
        np.clip(q[self.dof_ids], *self.get_robot_joint_range(), out=q[self.dof_ids])
        self.data.ctrl[self.actuator_ids] = q[self.dof_ids]

