from .ManipulatorBaseEnv import ManipulatorBaseEnv
from domain_object.builder import DomainObject
import numpy as np
from print_color import print
import mujoco


class ManipulatorGraspEnv(ManipulatorBaseEnv):
    def __init__(self, domain_object: DomainObject):
        super().__init__(domain_object)
        self.config_env         = domain_object.config_env
        self.reaching_threshold = domain_object.config_env.reaching_threshold
        self.finger_reach       = False
        # ----
        self._finger_velocity_buffer = []
        self._finger_grasped         = False
        self.set_params()

    def set_params(self):
        self.__set_ids()

    def check_local_hand_frame(self):
        id_g = self.model.site("canonical_fingertip_surafece").id
        R    = self.data.site_xmat[id_g].reshape(3, 3)
        z_axis_world = R[:, 2]   # ローカル z
        print(f"z_axis_world = [{z_axis_world[0]:.3f}, {z_axis_world[1]:.3f}, {z_axis_world[2]:.3f}]")
        # import ipdb; ipdb.set_trace()
        # import sys
        # sys.exit()


    def __set_ids(self):
        self.id_geom_right_fingertip = self.geom.name2id("right_fingertip_center")
        self.id_geom_left_fingertip  = self.geom.name2id("left_fingertip_center")
        self.id_object               = self.geom.name2id("object")
        self.id_finger_actuator      = self.model.actuator("actuator_finger").id
        self.id_joint_finger         = self.model.joint("finger_joint1").id
        self.id_body_object          = self.body.name2id("textured")
        self.id_body_hand            = self.body.name2id("hand")
        self.id_site_canonical       = self.site.name2id("canonical_fingertip_surafece")
        self.id_canonical_origin     = self.site.name2id("canonical_fingertip_origin")
        # ----
        self.id_end_effector_site = self.site.name2id(self.config_env.end_effector_site_name)
        # ----
        # import ipdb; ipdb.set_trace()
        self.print_object_mass()


    def get_offset_pose(self):
        self.R_WC = self.data.site(self.id_canonical_origin).xmat.reshape(3, 3)
        self.t_WC = self.data.site(self.id_canonical_origin).xpos.copy()
        # import ipdb; ipdb.set_trace()

    def palm_site_xpos(self):
        return self.site.xpos(site_id=self.id_end_effector_site)

    def object_xpos(self):
        return self.body.xpos(body_id=self.id_body_object)

    def fingertip_center_xpos(self):
        right_xpos  = self.geom.xpos(geom_id=self.id_geom_right_fingertip)
        left_xpos   = self.geom.xpos(geom_id=self.id_geom_left_fingertip)
        center_xpos = 0.5 * (right_xpos + left_xpos)
        return center_xpos

    def print_object_mass(self):
        object_body_mass = self.model.body_mass[self.id_body_object]
        print(f"object_body_mass = {object_body_mass:.3f} [kg]", tag="env", color="y", tag_color="y")

    def set_qpos_object(self, qpos_object: np.ndarray):
        assert qpos_object.shape == (7,) # pos + quat
        self.data.qpos[-7:] = qpos_object

    def set_qpos_palm(self, qpos_palm: np.ndarray):
        assert qpos_palm.shape == (7,) # pos + quat
        self.data.qpos[:7] = qpos_palm

    def set_qpos_finger(self, qpos_finger: np.ndarray):
        self.data.qpos[7:7+2] = qpos_finger

    def set_ctrl_finger(self, ctrl_finger: np.ndarray):
        self.data.ctrl[self.id_finger_actuator] = ctrl_finger
        # self.model.actuator("actuator_finger")

    def reset_finger_reach_flag(self):
        self.finger_reach = False

    def check_finger_reach(self, desired_finger_pos: float):
        """
        開閉コマンドを送った後、十分に指が動かなくなったか確認するための例。

        :param desired_finger_pos: 目標の指の角度または開閉幅
        :param finger_joint_ids:   グリッパ指のjoint_idのリスト
        :param pos_thresh:         位置誤差の閾値 (例: 0.005 rad 〜 数 mm 程度)
        :param vel_thresh:         速度閾値 (例: 0.001 rad/s)
        :return: bool (到達したかどうか)
        """
        # 1) 指の現在位置と目標位置の誤差
        finger_current_pos = self.data.qpos[self.id_joint_finger]
        error_finger       = np.linalg.norm(desired_finger_pos - finger_current_pos)
        # 2) 指の現在速度
        finger_current_vel = self.data.qvel[self.id_joint_finger]
        speed_finger       = np.linalg.norm(finger_current_vel)
        # 3) 判定ロジック
        reached_pos = (error_finger < self.reaching_threshold.finger.pos)
        slowed_down = (speed_finger < self.reaching_threshold.finger.vel)
        # ----
        if reached_pos and slowed_down:
            self.finger_reach = True


    def check_finger_stop(self):
        # すでに把持完了フラグがTrueの場合は何もしない
        if self._finger_grasped:
            return True

        # 現在の指ジョイント速度を取得 (例: data.qvel から finger_joint_ids を抜き出す)
        finger_joint_vel = self.data.qvel[self.id_joint_finger]
        finger_speed     = np.linalg.norm(finger_joint_vel)
        # print(f"finger_speed = {finger_speed}")

        # MuJoCoの時間刻み (制御周期)
        dt = self.model.opt.timestep  # or your own control_dt if separate

        # 速度が小さいかどうかをバッファに記録(1:停止近い, 0:動いてる)
        if finger_speed < self.reaching_threshold.finger.velocity:
            self._finger_velocity_buffer.append(1)
        else:
            self._finger_velocity_buffer.append(0)

        # バッファの長さを、stable_time に相当するステップ数までに限定
        max_buffer_len = int(np.ceil(self.reaching_threshold.finger.stable_time / dt))
        if len(self._finger_velocity_buffer) > max_buffer_len:
            self._finger_velocity_buffer.pop(0)

        # バッファ内がすべて「1」(=停止近い) なら stable_time 続けて停止しているとみなす
        if len(self._finger_velocity_buffer) == max_buffer_len:
            if all(self._finger_velocity_buffer):
                # 連続安定停止 → 把持完了！
                self._finger_grasped = True

        return self._finger_grasped
