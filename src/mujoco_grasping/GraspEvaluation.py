import os
import mujoco
import numpy as np
from domain_object.builder import DomainObject
from robot_env.instance.arm.ManipulatorGraspEnv import ManipulatorGraspEnv
from service import is_within_orientation_threshold
from print_color import print


class GraspEvaluation:
    def __init__(self, domain_object: DomainObject):
        self.geom = domain_object.geom
        self.body = domain_object.body
        self.env : ManipulatorGraspEnv   = domain_object.env
        self.grasp_pos_threshold = domain_object.config_grasp_evaluation.pos_threshold
        self.grasp_ori_threshold = np.deg2rad(domain_object.config_grasp_evaluation.ori_threshold_deg)
        self.push_threshold_z    = domain_object.config_grasp_evaluation.push_threshold_z
        # ----
        self.offset_for_mujoco   = domain_object.offset_for_mujoco
        self.z_direction_world   = domain_object.z_direction_world
        self.lift_up_height      = domain_object.lift_up_height
        # ----
        # 安全違反が起きたかどうかのフラグ
        self.safety_violation    = False
        # ----
        self.results_save_dir    = domain_object.results_save_dir
        # ----
        self.set_target_and_initial_pos()

    def set_target_and_initial_pos(self):
        id_table_geom     = self.geom.name2id("table")
        table_geom_size   = self.env.model.geom_size[id_table_geom]
        table_geom_height = (2 * table_geom_size[2])

        # 把持前のテーブル上の初期位置
        init_object_pos      = (self.z_direction_world * table_geom_height) + self.offset_for_mujoco
        self.init_object_pos = init_object_pos.copy()

        # リフトアップ後のターゲット位置
        add_lift_up_pos = (self.lift_up_height * self.z_direction_world)
        self.target_pos = init_object_pos + add_lift_up_pos

    def set_target_quat(self):
        object_body_id   = self.body.name2id("textured")
        self.target_quat = self.body.xquat(body_id=object_body_id)

        print(f"target_quat = [{self.target_quat[0]:.2f}, {self.target_quat[1]:.2f},  {self.target_quat[2]:.2f}, {self.target_quat[3]:.2f}]")
        # import ipdb; ipdb.set_trace()


    # ==== 追加: z 方向押し込み量の計算 ====
    def _compute_z_push_distance(self, current_pos: np.ndarray) -> float:
        """
        初期位置から world z 軸方向にどれだけ『下向き』に押し込んだか [m] を返す。
        z_direction_world は「上向き」単位ベクトルを想定。
        """
        disp                = (current_pos - self.init_object_pos)
        signed_disp_along_z = np.dot(disp, self.z_direction_world)
        push_towards_table  = max(0.0, -signed_disp_along_z)
        return float(push_towards_table)

    def is_pushed_excessively(self, current_pos: np.ndarray | None = None) -> bool:
        """
        world z 軸方向に push_threshold_z 以上押し込んだら True を返し，
        そのとき safety_violation フラグを立てる。
        """
        if current_pos is None:
            current_pos = self.env.data.body(self.env.id_body_object).xpos

        push_depth = self._compute_z_push_distance(current_pos)
        pushed = push_depth > self.push_threshold_z
        # print("push_depth = ", push_depth)

        if pushed:
            self.safety_violation = True
            print(
                f"[Safety] Object pushed along z: {push_depth:.3f} m "
                f"(threshold {self.push_threshold_z:.3f} m)",
                tag="eval", color="red", tag_color="red"
            )

        return pushed

    def reset_safety_flag(self):
        """1 試行ごとに呼んで safety_violation をリセットする用（任意）。"""
        self.safety_violation = False


    def evaluate(self, save: bool=False):
        if self.safety_violation:
             # もしループ中に安全違反が起きていたら、ここで即失敗
            grasped = False
        else:
            current_pos  = self.env.data.body(self.env.id_body_object).xpos
            current_quat = self.env.data.body(self.env.id_body_object).xquat
            # -----
            error_pos   = (self.target_pos - current_pos)
            # import ipdb; ipdb.set_trace()
            # ---------------- Orientation error --------------------
            current_quat_conj = np.zeros(4)
            error_quat        = np.zeros(4)
            mujoco.mju_negQuat(current_quat_conj, current_quat) # Conjugate quaternion (opposite same rotation)
            mujoco.mju_mulQuat(error_quat, self.target_quat, current_quat_conj)
            # --------------------------------------------------------------------
            grasped_position = (np.linalg.norm(error_pos)  < self.grasp_pos_threshold)
            grasped_quat     = is_within_orientation_threshold(error_quat, self.grasp_ori_threshold)
            grasped          = bool(grasped_position and grasped_quat)

        # -------------------------------------
        print("grasped =", grasped, tag="eval", color="yellow", tag_color="yellow")
        # --------------
        if save:
            np.save(os.path.join(self.results_save_dir, "grasp_success.npy"), grasped)
            np.savetxt(fname=os.path.join(self.results_save_dir, "grasp_success.txt"), X=[int(grasped)], fmt="%d")
        # --------------
        return grasped
