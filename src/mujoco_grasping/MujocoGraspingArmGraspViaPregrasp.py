import os
import numpy as np
from domain_object.builder import DomainObject
from robot_env.instance.arm.ManipulatorGraspEnv import ManipulatorGraspEnv
from robot_env.instance.utils.compute_qpos_finger_for_antipodal_gripper import compute_qpos_finger_for_antipodal_gripper
from robot_env.utils import save_video, save_captured_frame
from .sub_module import Do_FingerReach, Do_StayHere
from .sub_module import Do_OptimalGraspPoseReaching
from .sub_module import Do_PreGraspPoseReaching
from .sub_module import Do_LiftUp
from .utils.compute_pregrasp_from_grasp import compute_pregrasp_from_grasp
from value_object import ISFResult
from service import ExtendedRotation
from copy import deepcopy
from print_color import print
from .FrameCapture import FrameCapture


class MujocoGraspingArmGraspViaPregrasp:
    def __init__(self, domain_object: DomainObject):
        self.geom = domain_object.geom
        self.body = domain_object.body
        self.robot_name = domain_object.robot_name
        self.viewer_wrapper                 = domain_object.viewer_wrapper
        self.env : ManipulatorGraspEnv      = domain_object.env
        self.grasp_evaluator                = domain_object.grasp_evaluator
        self.qpos_object                    = domain_object.qpos_object
        self.d0                             = domain_object.d0
        self.d_min                          = domain_object.d_min
        self.d_bias                         = domain_object.d_bias
        self.object_whole_surface           = domain_object.object_whole_surface
        # ----
        self.show_ui                        = domain_object.show_ui
        self.config_viewer                  = domain_object.config_env.viewer
        self.config_env                     = domain_object.config_env
        self.stay_step                      = domain_object.config_ik_solver.stay_step
        # ----
        self.n_z0                           = domain_object.n_z0
        self.sphere_radius                  = domain_object.sphere_radius
        self.lift_up_height                 = domain_object.lift_up_height
        self.z_direction_world              = domain_object.z_direction_world
        # ---
        self.model_name                     = domain_object.model_name
        # ----
        self.results_save_dir               = domain_object.results_save_dir
        self.grasp_evaluator                = domain_object.grasp_evaluator
        # ----
        self.do_pre_grasp                   = Do_PreGraspPoseReaching(domain_object)
        self.do_optimal_grasp               = Do_OptimalGraspPoseReaching(domain_object)
        self.do_finger_reach                = Do_FingerReach(domain_object)
        self.do_lift_up                     = Do_LiftUp(domain_object)
        self.do_stay_here                   = Do_StayHere(domain_object)
        # ----
        self.frame_capture                  = FrameCapture(domain_object)
        # ----
        self.object_mujoco_load_pos        = domain_object.object_mujoco_load_pos
        self.object_mujoco_load_quat       = domain_object.object_mujoco_load_quat
        # -----
        self._set_table_hight()
        self._set_object_pos()
        self._set_object_point_cloud_z_offset()

    def _set_table_hight(self):
        id_table_geom    = self.geom.name2id("table")
        table_geom_size  = self.env.model.geom_size[id_table_geom]
        table_geom_hight = (2 * table_geom_size[2])
        self.table_hight = table_geom_hight

    def _set_object_pos(self):
        # id_object_body        = self.body.name2id("textured")
        # object_body_pos_xml   = self.env.model.body_pos[id_object_body]
        # ---
        self.object_pos     = deepcopy(self.object_mujoco_load_pos)
        self.object_pos[-1] += self.table_hight

    def _set_object_point_cloud_z_offset(self):
        self.object_point_cloud_z_offset = min(self.object_whole_surface.points[:, -1])
        print(f"{self.object_point_cloud_z_offset}",
                        tag = "Grasp Z-offset", color="y", tag_color="y")

    def add_table_height(self, qpos_palm):
        trans_add_table_hight = (self.z_direction_world * self.table_hight)
        return (qpos_palm + trans_add_table_hight)

    def execute(self, isf_result: ISFResult):
        # ===================== environment setting =====================
        self.env.set_home_keyframe(key_name="home", object_pos=self.object_pos, object_quat=self.object_mujoco_load_quat)
        self.env.reset(keyframe=0)
        self.env.set_params()
        self.env.check_local_hand_frame()
        self.env.get_offset_pose() # <- need for ISF execution in Mujoco
        self.grasp_evaluator.set_target_quat()

        # ================ ISF planning result conversion ================
        # canonical gripper surface ∂𝐹 の結果
        # *canonical gripper surface:
        #   「グリッパが閉じたとき，左右の平面の中央位置が原点 (0,0,0) で、
        #     回転なしでz-upの右手座標系。x軸は原点から左指に伸びる方法」
        '''
            CGSW = Canonical Gripper Suraface on World coordinate
        '''
        rotation_gripper_frame    = isf_result.rotation    # canonical gripper の world 回転
        translation_gripper_frame = isf_result.translation # canonical gripper の world 並進
        delta_d                   = isf_result.delta_d     # 開閉幅

        # --------------- apply offset pose of hand ---------------
        R_G = rotation_gripper_frame.as_matrix()
        t_G = translation_gripper_frame
        # ------
        R_WG_opt    = self.env.R_WC @ R_G
        t_WG_opt    = self.env.t_WC + (self.env.R_WC @ t_G)
        quat_WG_opt = ExtendedRotation.from_matrix(R_WG_opt).as_quat_scalar_first()

        # ------------------- pre-grasp -------------------
        _, t_WG_pre = compute_pregrasp_from_grasp(R_WG_opt, t_WG_opt, self.sphere_radius)
        quat_WG_pre = quat_WG_opt.copy()
        # ------------------ lift up pose -----------------
        t_WG_liftup    = t_WG_opt + (self.lift_up_height * self.z_direction_world)
        quat_WG_liftup = quat_WG_opt.copy()

        # =================== compute qpos finger ===================
        qpos_finger = compute_qpos_finger_for_antipodal_gripper(
            d0      = self.d0,
            d_min   = self.d_min,
            delta_d = delta_d,
            d_bias  = self.d_bias,
        )
        # import ipdb; ipdb.set_trace()

        # =================== launch viewer ===================
        self.viewer_wrapper.launch()
        self.viewer_wrapper.initialize_for_env()

        # =================== do grasping ===================
        # 2) with ブロックで使う
        with self.viewer_wrapper as viewer:




            viewer.camera.set_overview()

            # ------ data capture setup for paper ------
            # plese delete after debug
            gv = viewer._gui_viewer
            with gv.lock():
                gv.cam.lookat[:]  = np.array([-0.25, 0, 0.35] )
                gv.cam.distance   = 0.5
                gv.cam.azimuth    = 230.0
                gv.cam.elevation  = -20.0
            # -----------------------------------------


            self.frame_capture.home(frame=viewer.sync())
            import ipdb; ipdb.set_trace()
            # ----- (1) pregrasp -----
            self.do_pre_grasp.execute(viewer, t_WG_pre, quat_WG_pre)
            self.do_stay_here.execute(viewer, stay_step=self.stay_step.pre_grasp)
            self.frame_capture.pregrasp(frame=viewer.sync())
            # ----- (2) grasp -----
            self.do_optimal_grasp.execute(viewer, t_WG_opt, quat_WG_opt)
            self.do_stay_here.execute(viewer, stay_step=self.stay_step.optimal_reach)
            # ----- (3) gripper close  -----
            self.do_finger_reach.execute(viewer, qpos_finger)
            self.do_stay_here.execute(viewer, stay_step=self.stay_step.finger_close)
            self.frame_capture.grasp(frame=viewer.sync())
            # ----- (4) postgrasp  -----
            self.do_lift_up.execute(viewer, t_WG_liftup,  quat_WG_liftup)
            self.do_stay_here.execute(viewer, stay_step=self.stay_step.lift_up)
            self.frame_capture.postgrasp(frame=viewer.sync())

            # ===================================================================================
            self.grasp_evaluator.evaluate(save=True)
            # ===================================================================================

            if not self.config_env.viewer.use_gui:
                # -------
                save_video(
                    frames    = viewer.frames,
                    fps       = self.config_viewer.save.fps,
                    skip      = self.config_viewer.save.skip,
                    save_path = os.path.join(self.results_save_dir,  self.config_viewer.save.filename + f"_{self.model_name}" +".mp4"),
                )
                # -------
                save_captured_frame(
                    frame = viewer.sync(),
                    save_path = os.path.join(self.results_save_dir, "lift_up_overview" + f"_{self.model_name}" +".png"),
                )

                fingertip_center = self.env.fingertip_center_xpos()
                viewer.camera.set_zoom_with_fingertip_center(fingertip_center=fingertip_center)
                # viewer.camera.set_zoom_with_fingertip_center(fingertip_center= self.env.fingertip_center_xpos())
                save_captured_frame(
                    frame = viewer.sync(),
                    save_path = os.path.join(self.results_save_dir, "lift_up_zoom" + f"_{self.model_name}" +".png"),
                )
        # -------

