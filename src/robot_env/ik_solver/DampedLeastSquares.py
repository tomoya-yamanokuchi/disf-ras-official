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
        # 制御周期（シミュレーションの timestep と揃える想定）
        self.integration_dt = config_ik_solver.integration_dt
        # DLS のダンピング係数
        self.damping        = config_ik_solver.damping
        # ジョイント角速度の上限（rad/s）
        self.max_angvel     = config_ik_solver.max_angvel
        # Task-space proportional gains
        self.kp_pos        = getattr(config_ik_solver, 'kp_pos', 4.0)
        self.kp_ori        = getattr(config_ik_solver, 'kp_ori', 4.0)
        # ---
        # 位置・姿勢の到達判定用の閾値（絶対誤差）
        self.reaching_threshold = config_ik_solver.reaching_threshold

        # --- 相対的な収束判定用のパラメータ --------------------------  # NEW
        # 誤差ノルムの「速度」(1秒あたり) がこの値以下なら「ほぼ止まっている」とみなす
        self.pos_error_vel_threshold = getattr(
            config_ik_solver, 'pos_error_vel_threshold', 1e-3
        )
        self.ori_error_vel_threshold = getattr(
            config_ik_solver, 'ori_error_vel_threshold', 1e-3
        )
        # 初期誤差に対する相対誤差の閾値（例: 0.05 → 初期の5%以下まで来たらOK候補）
        self.pos_error_rel_ratio = getattr(
            config_ik_solver, 'pos_error_rel_ratio', 0.05
        )
        self.ori_error_rel_ratio = getattr(
            config_ik_solver, 'ori_error_rel_ratio', 0.05
        )
        # ターゲット更新直後は何ステップか必ず動く (初期状態と停止状態の区別用)
        self.min_reach_steps = getattr(
            config_ik_solver, 'min_reach_steps', 10
        )
        # 上の条件を満たした状態が何ステップ連続したら reached とみなすか
        self.stable_steps_required = getattr(
            config_ik_solver, 'stable_steps_required', 5
        )
        # --------------------------------------------------------------  # /NEW

        # ---
        if config_ik_solver.gravity_compensation:
            self.gravity_compensation_setting()
        # ---
        self.target_pos  = None
        self.target_quat = None
        # ---
        self.reached     = False

        # 誤差履歴の初期化                                        # NEW
        self._reset_error_history()

    def _reset_error_history(self):                               # NEW
        """相対的な収束判定用の履歴をリセット"""
        self.pos_error_norm_prev  = None
        self.ori_error_norm_prev  = None
        self.initial_pos_error_norm = None
        self.initial_ori_error_norm = None
        self.steps_since_target_update = 0
        self.stable_counter = 0

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
        self.reached     = False                     # NEW
        self._reset_error_history()                  # NEW
        # import ipdb; ipdb.set_trace()

    def update_current_pose(self):
        self.current_pos  = self.data.site(self.site_id).xpos
        self.current_xmat = self.data.site(self.site_id).xmat # rotation info
        # print("current rotvec =", np.rad2deg(ExtendedRotation.from_matrix(self.current_xmat.reshape(3, 3)).as_rotvec()))
        # import ipdb; ipdb.set_trace()

    def reset_ik_reach_flag(self):
        self.reached = False
        self._reset_error_history() # NEW

    def solve(self):
        """Resolved-rate DLS IK step towards (target_pos, target_quat).

        target_pos / target_quat はユーザ側で update_target_pose() 等から事前にセットしておく想定。
        """
        assert self.target_pos is not None
        assert self.target_quat is not None
        '''
        target_pos and target_quat should be updated by user from usecase
        '''

        # 現在のエンドエフェクタ姿勢を取得
        self.update_current_pose()

        # --- 誤差計算 ----------------------------------------------------
        # 位置誤差
        self.error_pos[:] = (self.target_pos - self.current_pos)

        # 姿勢誤差（target * current_conj を角速度として表現）
        mujoco.mju_mat2Quat(self.current_quat, self.current_xmat)
        mujoco.mju_negQuat(self.current_quat_conj, self.current_quat)  # Conjugate quaternion
        mujoco.mju_mulQuat(self.error_quat, self.target_quat, self.current_quat_conj)
        mujoco.mju_quat2Vel(self.error_ori, self.error_quat, 1.0)

        # import ipdb; ipdb.set_trace()

        # --- ヤコビアン --------------------------------------------------
        mujoco.mj_jacSite(self.model, self.data, self.jac[:3], self.jac[3:], self.site_id)

        # --- タスク空間速度 dx を構成（ここでゲインを入れる） ----------
        dx = np.empty(6)
        dx[:3] = self.kp_pos * self.error_pos
        dx[3:] = self.kp_ori * self.error_ori

        # --- Damped Least Squares: (J J^T + λI) v = dx, dq = J^T v ------
        v  = np.linalg.solve(self.jac @ self.jac.T + self.diag, dx)
        dq = self.jac.T @ v

        # --- 速度制限 ----------------------------------------------------
        if self.max_angvel > 0:
            dq_abs_max = np.abs(dq).max()
            if dq_abs_max > self.max_angvel:
                dq *= self.max_angvel / dq_abs_max

        # --- 速度を積分して関節位置を更新 ------------------------------
        q = self.data.qpos.copy()
        mujoco.mj_integratePos(self.model, q, dq, self.integration_dt)

        # 関節レンジでクリップして ctrl に反映
        np.clip(q[self.dof_ids], *self.get_robot_joint_range(), out=q[self.dof_ids])
        self.data.ctrl[self.actuator_ids] = q[self.dof_ids]

        # --- 収束判定 ----------------------------------------------------
        pos_error_norm = np.linalg.norm(self.error_pos)
        ori_error_norm = np.linalg.norm(self.error_ori)

        # 初期誤差を記録（ターゲット更新後の「最初の solve 呼び出し」でセット）   # NEW
        if self.initial_pos_error_norm is None:
            self.initial_pos_error_norm = pos_error_norm
            self.initial_ori_error_norm = ori_error_norm

        # 誤差ノルムの変化量から「速度」（1秒あたり）を近似                  # NEW
        if self.pos_error_norm_prev is None:
            pos_error_vel = np.inf
            ori_error_vel = np.inf
        else:
            pos_error_vel = abs(pos_error_norm - self.pos_error_norm_prev) / self.integration_dt
            ori_error_vel = abs(ori_error_norm - self.ori_error_norm_prev) / self.integration_dt

        # 次のステップのために保存                                         # NEW
        self.pos_error_norm_prev = pos_error_norm
        self.ori_error_norm_prev = ori_error_norm
        self.steps_since_target_update += 1

        # --- 条件1: 絶対誤差が小さいか ---------------------------------
        abs_pos_ok = (pos_error_norm < self.pos_threshold)
        abs_ori_ok = (ori_error_norm < self.ori_threshold)

        # --- 条件2: 初期誤差に対して十分小さくなっているか --------------
        # （絶対値だけだとスケール依存になるので、念のため相対条件も）      # NEW
        rel_pos_ok = (pos_error_norm < self.pos_error_rel_ratio * self.initial_pos_error_norm)
        rel_ori_ok = (ori_error_norm < self.ori_error_rel_ratio * self.initial_ori_error_norm)

        # --- 条件3: 誤差がほとんど変化していない（停止している） ---------
        vel_pos_ok = (pos_error_vel < self.pos_error_vel_threshold)
        vel_ori_ok = (ori_error_vel < self.ori_error_vel_threshold)

        # --- 初期状態と停止状態の区別のためのヒステリシス ---------------
        # ・ターゲット更新から十分ステップが経過している
        # ・絶対誤差 & 相対誤差が小さい
        # ・誤差の変化速度も小さい
        if (self.steps_since_target_update >= self.min_reach_steps and
            abs_pos_ok and abs_ori_ok and
            rel_pos_ok and rel_ori_ok and
            vel_pos_ok and vel_ori_ok):
            self.stable_counter += 1
        else:
            self.stable_counter = 0

        if self.stable_counter >= self.stable_steps_required:
            self.reached = True

