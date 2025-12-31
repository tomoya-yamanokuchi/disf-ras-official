import os
import numpy as np
from copy import deepcopy
from print_color import print

class SelfContainedDomainObjectBuilder:
    """
        This object is a "Self-contained Domain Object Builder",
        where the object itself has both of the role of
        instantiation of objects and the role as a usual Domain Object.

        This means that this object looks like a
            (1) "bulder object" for director object
        while it looks like a
            (2) "domain object" for the external object
                    that receives this object in its constructor.
    """

    def get_domain_object(self):
        return self

    def set_current_working_dir(self):
        import os
        import pathlib
        self.cwd_path = pathlib.Path(os.getcwd())

    def set_robot_name(self, robot_name: str):
        self.robot_name = robot_name

    def set_object_name(self, object_name: str):
        self.object_name = object_name

    # ===================== Mujoco Env =====================
    def build_model(self, load_xml_path: str = None):
        import mujoco

        if load_xml_path is None:
            load_xml_path = self.cwd_path.joinpath("assets", self.robot_name, self.output_file_name)
        # ---
        # print("-----------------")
        # print("env_xml_path = ", load_xml_path)
        # print("-----------------")
        print(load_xml_path,
                tag = "XML Path", color="c", tag_color="c")
        # ---
        # import ipdb; ipdb.set_trace()
        self.model = mujoco.MjModel.from_xml_path(str(load_xml_path))
        self.model.opt.timestep = self.config_env.option.timestep # Override the simulation timestep.

    def buid_data(self):
        import mujoco
        self.data = mujoco.MjData(self.model)

    def buid_body(self):
        from robot_env.utils import BodyManager
        self.body = BodyManager(self.model, self.data)

    def buid_site(self):
        from robot_env.utils import SiteManager
        self.site = SiteManager(self.model, self.data)

    def build_geom(self):
        from robot_env.utils.GeomManager import GeomManager
        self.geom = GeomManager(self.model, self.data)

    def build_camera(self):
        from robot_env.utils.Camera import Camera
        self.camera = Camera(self.config_env, model=self.model)

    def build_option(self):
        from robot_env.utils.Option import Option
        self.option = Option(config_viewer=self.config_env.viewer)

    def build_scene(self):
        from robot_env.utils.Scene import Scene
        self.scene = Scene(self.config_env, model=self.model)

    def build_viewer(self):
        from robot_env.utils import MyViewerWrapper
        self.viewer_wrapper = MyViewerWrapper(
            model      = self.model,
            data       = self.data,
            camera     = self.camera,
            opt        = self.option.option,
            scn        = self.scene.scene,
            config_env = self.config_env
        )

    def build_config_env(self, config_name: str = "panda_hand"):
        from config_loader import ConfigLoader

        # :::::::::::: domain object ::::::::::::
        self.config_env = ConfigLoader.load_env(
            config_name = config_name,
        )
        # ----
        self.sphere_radius     = self.config_env.pre_grasp.sphere_radius
        self.lift_up_height    = self.config_env.pre_grasp.lift_up_height
        self.z_direction_world = np.array(self.config_env.pre_grasp.z_direction_world)
        self.show_ui           = self.config_env.viewer.show_ui
        self.d_bias            = self.config_env.d_bias

    def build_config_ik_solver(self, config_name: str = "default_ik_solver"):
        from config_loader import ConfigLoader
        self.config_ik_solver = ConfigLoader.load_ik_solver(
            config_name = config_name,
        )

    def build_config_cma(self, config_name: str = "default"):
        from config_loader import ConfigLoader
        self.config_cma = ConfigLoader.load_cma(
            config_name = config_name,
        )


    def build_config_grasp_evaluation(self, config_name: str = "default"):
        from config_loader import ConfigLoader
        self.config_grasp_evaluation = ConfigLoader.load_grasp_evaluation(
            config_name = config_name,
        )

    def build_config_icp(self, config_name: str = "default_icp"):
        from config_loader import ConfigLoader
        self.config_icp = ConfigLoader.load_icp(
            config_name = config_name,
        )

    def build_config_gripper_surface(self, config_name: str = "default"):
        from config_loader import ConfigLoader
        self.config_gripper_surface = ConfigLoader.load_gripper_surface(
            config_name = config_name,
        )

    def build_config_rotation_initialization(self, config_name: str = "default_rot_init"):
        from config_loader import ConfigLoader
        self.config_rot_init = ConfigLoader.load_rot_init(
            config_name = config_name,
        )

    def build_env(self):
        from robot_env import RobotEnvFactory
        self.env = RobotEnvFactory.create(
            env_name      = self.config_env.env_name,
            domain_object = self,
        )


    def build_ik_solver(self):
        from robot_env.ik_solver import DampedLeastSquares
        self.ik_solver = DampedLeastSquares(
            model            = self.model,
            data             = self.data,
            config_env       = self.config_env,
            config_ik_solver = self.config_ik_solver,
        )

    def build_grasp_evaluator(self):
        from mujoco_grasping import GraspEvaluation
        self.grasp_evaluator = GraspEvaluation(self)

    # ======================== ISF config ========================
    def build_config_isf(self, config_name: str):
        from config_loader import ConfigLoader
        # :::::::::::: domain object ::::::::::::
        self.config_isf = ConfigLoader.load_isf(
            config_name = config_name,
        )


    def build_ipfo_parameters(self):
        from service import normalize_vector
        from service import compute_object_qpos
        # :::::::::::: domain object ::::::::::::
        # ----------------------------------------
        self.n_z0 = normalize_vector(self.config_isf.hand_plane_z)
        self.v0   = normalize_vector(self.config_isf.gripper_normal)

        self.alpha              = self.config_isf.alpha
        self.beta               = self.config_isf.beta
        # self.gamma              = self.config_isf.gamma

        self.rotvec_rad_object  = np.deg2rad(self.config_isf.object_pose.rotvec_degree)
        self.translation_object = np.array(self.config_isf.object_pose.translation)

        # import ipdb; ipdb.set_trace()
        # self.num_fingertip_surface_points = self.config_gripper_surface.num_fingertip_surface_points

        self.d_min                        = self.config_gripper_surface.d_min
        self.d_max                        = self.config_gripper_surface.d_max
        self.d0                           = self.config_gripper_surface.d0

        self.delta_e            = self.config_isf.delta_e

        self.gripper_z_width = self.config_gripper_surface.gripper_z_width
        self.gripper_x_width = self.config_gripper_surface.gripper_x_width

        self.offset_for_mujoco  = np.array(self.config_isf.gripper_pose.offset_for_mujoco)
        # self.delta_d_est_init   = self.config_isf.delta_d_est_init
        # self.delta_d_opt        = (self.d0 - self.object_width)
        self.epsilon            = self.config_isf.epsilon
        self.verbose            = self.config_isf.verbose
        self.method_name        = self.config_isf.method_name
        # -----------
        self.qpos_object = compute_object_qpos(**self.config_isf.object_pose)
        # -----------
        self.save_dir = self.config_isf.save.save_dir
        from service import create_directory_if_not_exists
        create_directory_if_not_exists(directory=self.save_dir)



    def build_ipfo_text_logger(self):
        from grasp_planning.utils import IPFOTextLogger
        self.text_logger = IPFOTextLogger(self)

    def build_surface_visualizer(self):
        from grasp_planning.disf.visualization import SurfaceVisualization
        self.surface_visualizer = SurfaceVisualization(self)

    def build_loop_criteria(self):
        from grasp_planning.utils import LoopCriteria
        self.loop_criteria = LoopCriteria(self)

    def build_isf_loop_criteria(self):
        from grasp_planning.utils import ISF_LoopCriteria
        self.isf_loop_criteria = ISF_LoopCriteria(self)

    def build_config_point_cloud_data(self, config_name: str):
        from config_loader import ConfigLoader
        self.config_pc_data = ConfigLoader.load_point_cloud_data(
            config_name = config_name,
        )
        # ----
        self.gripper_translation        = np.array(self.config_pc_data.grasp[self.robot_name].gripper_pose.translation)
        self.gripper_rotvec             = np.array(self.config_pc_data.grasp[self.robot_name].gripper_pose.rotvec)
        # ----
        from service import normalize_vector
        n_app_unnorm = np.array(self.config_pc_data.grasp[self.robot_name].n_approach)
        self.n_app   = normalize_vector(n_app_unnorm)
        # ------------------------------------------------------
        self.model_name                       = self.config_pc_data.model_name
        self.output_file_name                 = self.config_pc_data.generate.output_file_name
        self.pc_data_dir_path_relative_to_cwd = self.config_pc_data.load.dir_path_relative_to_cwd
        self.suffix_fname                     = self.config_pc_data.load.suffix_fname
        self.rotvec_object_in_pcd_load        = np.array(self.config_pc_data.load.object_frame_to_gripper_frame_transform.rotvec_object_in_pcd_load)
        self.translation_object_in_pcd_load   = np.array(self.config_pc_data.load.object_frame_to_gripper_frame_transform.translation_object_in_pcd_load)
        self.object_mujoco_load_pos           = np.array(self.config_pc_data.load.object_mujoco_load.pos)
        self.object_mujoco_load_quat          = np.array(self.config_pc_data.load.object_mujoco_load.quat)
        # -------------------------------------------------------
        from service import create_directory_if_not_exists
        #  self.results_save_dir = os.path.join(self.save_dir, self.method_name, self.model_name)
        self.results_save_dir = os.path.join(self.save_dir, self.robot_name, self.method_name, self.model_name)

        create_directory_if_not_exists(self.results_save_dir)


    def build_ycb_xml_scene_file(self, robot_name: str, object_name: str):
        from xml_generation import generate_mujoco_scene
        # ----

        # import ipdb; ipdb.set_trace()

        abs_base_path_for_xml = self.cwd_path.joinpath(self.pc_data_dir_path_relative_to_cwd)
        abs_robot_xml_path    = self.cwd_path.joinpath("assets", robot_name, f"{robot_name}.xml")
        xml_output_path       = self.cwd_path.joinpath("assets", robot_name, self.config_pc_data.generate.output_file_name)
        # ----
        # import ipdb; ipdb.set_trace()
        generate_mujoco_scene(
            output_file        = xml_output_path,
            # object_name        = object_name,
            # object_base_path   = str(abs_base_path_for_xml),

            obj_path = f"{str(abs_base_path_for_xml)}/{object_name}/tsdf/textured/textured.xml",

            robot_include_path = str(abs_robot_xml_path),
            robot_name         = robot_name,
            also_exclude_all_robot_vs_floor_table=True,  # set True if you want everything excluded vs table/floor
            table_size         = self.config_env.table.table_size,
        )
        # import ipdb; ipdb.set_trace()


    def build_custom_xml_scene_file(self, robot_name: str, object_name: str):
        from xml_generation import generate_mujoco_scene
        # ----


        abs_base_path_for_xml = self.cwd_path.joinpath(self.pc_data_dir_path_relative_to_cwd)

        model_name           = self.config_pc_data.model_name

        # import ipdb; ipdb.set_trace()

        abs_robot_xml_path    = self.cwd_path.joinpath("assets", robot_name, f"{robot_name}.xml")
        xml_output_path       = self.cwd_path.joinpath("assets", robot_name, self.config_pc_data.generate.output_file_name)
        # ----
        generate_mujoco_scene(
            output_file        = xml_output_path,
            # object_name        = object_name,
            # object_base_path   = str(abs_base_path_for_xml),
            # obj_path = f"{str(abs_base_path_for_xml)}/{model_dname}/{model_dname}/{model_dname}.xml",
            obj_path = f"{str(abs_base_path_for_xml)}/{model_name}/textured/textured.xml",

            robot_include_path = str(abs_robot_xml_path),
            robot_name         = robot_name,
            also_exclude_all_robot_vs_floor_table=True,  # set True if you want everything excluded vs table/floor
            table_size         = self.config_env.table.table_size,
        )


    def build_ycb_data_point_cloud(self):
        import open3d as o3d
        from point_cloud_loading.PointCloudLoader import PointCloudLoader
        pcd_loader = PointCloudLoader(self)
        pcd_loader.load()
        self.point_cloud = pcd_loader.get_point_cloud()


    def build_observed_pcd(self):
        import open3d as o3d
        from point_cloud_loading.ObservedPointCloudLoader import ObservedPointCloudLoader
        pcd_loader = ObservedPointCloudLoader(self)
        pcd_loader.load()
        self.point_cloud = pcd_loader.get_point_cloud()

    def build_custom_stl_data_point_cloud(self):
        import open3d as o3d
        from point_cloud_loading.CustomSTLPointCloudLoader import CustomSTLPointCloudLoader

        pcd_loader = CustomSTLPointCloudLoader(self)
        # pcd_loader.load(number_of_points=2000)
        pcd_loader.load(number_of_points=self.config_pc_data.load.number_of_points)
        pcd_loader.frame_transformation_with_R_and_t()
        pcd_loader.frame_transformation_with_z_offset_alignment()
        self.point_cloud = pcd_loader.get_point_cloud()




    def build_ycb_data_normal_estimation(self):
        import open3d as o3d
        # estimate normal vector
        self.point_cloud.estimate_normals(
            search_param = o3d.geometry.KDTreeSearchParamHybrid(
                radius = self.config_pc_data.pre_prosessing.normal_estimation.radius_neighbor_search,
                max_nn = self.config_pc_data.pre_prosessing.normal_estimation.max_neighbor_number,
            )
        )
        # adjust normal vector direction
        self.point_cloud.orient_normals_consistent_tangent_plane(
            k = self.config_pc_data.pre_prosessing.normal_estimation.k_normals_consistent,
        )
        # --- compute centroid of point cloud ---
        centroid = np.mean(np.asarray(self.point_cloud.points), axis=0)
        # --- adjust the direction of normal vector based on the centroid ---
        points   = np.asarray(self.point_cloud.points)
        normals  = np.asarray(self.point_cloud.normals)
        # # --- adjust the normal vector such that it faces outward ---
        for i in range(len(normals)):
            direction = points[i] - centroid       # vectro from centroid to points
            if np.dot(normals[i], direction) < 0:  # if normal vector is pointing at outside
                normals[i] = -normals[i]           # set it opposite direction
        # --- apply the aligned normal ---
        self.point_cloud.normals = o3d.utility.Vector3dVector(normals)





    def visualize_point_cloud_with_normals(self,
                                        normal_length: float = 0.01,
                                        max_num_normals: int = 2000):
        """
        self.point_cloud に格納された点群と法線ベクトルを同時に可視化する。

        Parameters
        ----------
        normal_length : float
            法線ベクトルの描画長さスケール。
        max_num_normals : int
            法線を表示する最大本数（多すぎると見づらいのでサンプリングする）。
        """
        import numpy as np
        import open3d as o3d

        if self.point_cloud is None:
            raise ValueError("self.point_cloud が None です。先に build_ycb_data_normal_estimation() を実行してください。")

        pcd = self.point_cloud

        # numpy array に変換
        points  = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)

        if len(points) == 0:
            raise ValueError("point cloud に点がありません。")

        # 法線本数が多すぎる場合はサンプリング
        num_points = len(points)
        if num_points > max_num_normals:
            step = num_points // max_num_normals
            indices = np.arange(0, num_points, step)
        else:
            indices = np.arange(num_points)

        sampled_points  = points[indices]
        sampled_normals = normals[indices]

        # LineSet 用の点と線インデックスを作成
        line_points = []
        lines = []
        colors = []

        for i, (p, n) in enumerate(zip(sampled_points, sampled_normals)):
            start = p
            end   = p + normal_length * n

            line_points.append(start)
            line_points.append(end)

            # 2 点ずつ追加しているので index は 2*i, 2*i+1
            lines.append([2 * i, 2 * i + 1])

            # 法線を赤色で表示（RGB）
            colors.append([1.0, 0.0, 0.0])

        line_points = np.asarray(line_points)
        lines       = np.asarray(lines, dtype=np.int32)
        colors      = np.asarray(colors)

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(line_points)
        line_set.lines  = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)

        # 点群を少しグレーにしておくと法線が見やすい
        pcd_for_vis = o3d.geometry.PointCloud(pcd)  # コピー
        pcd_for_vis.paint_uniform_color([0.7, 0.7, 0.7])

        # 可視化
        o3d.visualization.draw_geometries(
            [pcd_for_vis, line_set],
            window_name="Point Cloud with Normals",
            width=1280,
            height=720,
            point_show_normal=False,  # 自前で描画しているので False
        )



    def build_point_cloud_sdf(self):
        import open3d as o3d
        from point_cloud_loading.PointCloudSDF import PointCloudSDF

        self.pcd_sdf = PointCloudSDF(self.point_cloud)
        self.pcd_sdf.build_ycb_sdf()
        # self.pcd_sdf.visualize_sdf_slice(axis="z", index=19)

    def build_ycb_data_PointNormal_ValueObject(self):
        from value_object import PointNormalUnitPairs
        points  = np.asarray(self.point_cloud.points)
        normals = np.asarray(self.point_cloud.normals)
        self.ycb_point_normal = PointNormalUnitPairs(points, normals)
        # self.point_cloud = None

    def build_ycb_target_by_contact_estimation_with_hand_plane(self):
        from contact_detection import detect_plane_intersection
        from value_object import PointNormalUnitPairs

        # ★ whole-object のシフトはもうしないので、
        #   そのまま YCB ローカル（object frame）の点群を使う
        filtered_indices = detect_plane_intersection(
            point_cloud       = self.ycb_point_normal.points,
            hand_plane_origin = self.contact_plane_origin,
            hand_plane_normal = self.contact_plane_normal,
            d1                = self.d1,
            d2                = self.d2,
        )
        self.contact_indices = filtered_indices

        contact_points  = self.ycb_point_normal.points[filtered_indices]
        contact_normals = self.ycb_point_normal.normals[filtered_indices]

        # ★ 新しい名前：object_contact_surface
        self.object_contact_surface = PointNormalUnitPairs(
            points  = contact_points,
            normals = contact_normals,
        )

        # 互換性のために target だけは残してもよい（不要ならこれも削除可）
        self.target = self.object_contact_surface



    def build_ycb_target_by_contact_estimation_with_gripper_box(self):
        """
        手動で設定された gripper pose (self.gripper_rotvec, self.gripper_translation)
        と gripper サイズに基づいて、
        gripper inside box と交差する object surface を contact 候補として抽出する。
        """
        # SelfContainedDomainObjectBuilder.py の中
        from service import ExtendedRotation
        from value_object import PointNormalUnitPairs
        from contact_detection import detect_gripper_box_intersection_canonical

        # 1) gripper の姿勢を Rodrigues → 回転行列に変換
        #    （ISF の run() で使っているものと同じ）
        R0 = ExtendedRotation.from_rotvec(self.gripper_rotvec).as_rodrigues()  # 3x3
        t0 = self.gripper_translation  # (3,)

        # 2) object frame (YCB ローカル) → canonical gripper frame G へ逆変換
        #    p_G = R0^T (p_O - t0)
        P_O = self.ycb_point_normal.points  # (N,3)
        R_og = R0.T
        P_G = (R_og @ (P_O - t0).T).T       # (N,3), canonical gripper frame

        # 3) box サイズのパラメータを決定
        #    ここはプロジェクトの設定に応じて：
        fingertip_width   = self.gripper_x_width      # x方向 [例: mm]
        fingertip_height  = self.gripper_z_width      # z方向
        gripper_aperture  = self.d0                   # y方向: 現在の開口幅
        margin            = 0.0   # or 固定値 1.0 など

        # 4) canonical frame 上で box 判定 → contact 候補インデックス
        filtered_indices = detect_gripper_box_intersection_canonical(
            point_cloud_G    = P_G,
            fingertip_width  = fingertip_width,
            fingertip_height = fingertip_height,
            gripper_aperture = gripper_aperture,
            margin           = margin,
        )
        self.contact_indices = filtered_indices

        # 5) インデックスで元の object frame の点群を切り出す
        contact_points  = self.ycb_point_normal.points[filtered_indices]
        contact_normals = self.ycb_point_normal.normals[filtered_indices]

        self.object_contact_surface = PointNormalUnitPairs(
            points  = contact_points,
            normals = contact_normals,
        )

        # 互換性のため target も更新
        self.target = self.object_contact_surface




    def build_manual_target_by_contact_estimation_with_hand_plane(self):
        from contact_detection import detect_plane_intersection
        from value_object import PointNormalUnitPairs
        filtered_indices = detect_plane_intersection(
            point_cloud       = self.ycb_point_normal.points,
            hand_plane_origin = self.contact_plane_origin,
            hand_plane_normal = self.contact_plane_normal,
            d1                = self.d1,
            d2                = self.d2,
        )
        # ---
        self.contact_indices = filtered_indices
        contact_points       = self.ycb_point_normal.points[filtered_indices]
        contact_normals      = self.ycb_point_normal.normals[filtered_indices]
        # ----
        self.centered_ycb_contact_point_normal_by_whole =  PointNormalUnitPairs(
            points  = contact_points,
            normals = contact_normals,
        )

    def shift_ycb_target_center_to_origin(self):
        from service import shift_object_center_to_origin
        self.centered_ycb_contact_point_normal, self.t_shift_ycb_contact_by_contact = shift_object_center_to_origin(
            target_point_normal = self.centered_ycb_contact_point_normal_by_whole,
        )
        self.target                 = self.centered_ycb_contact_point_normal
        self.object_contact_surface = self.centered_ycb_contact_point_normal

    def shift_ycb_whole_object_center_to_origin(self):
        from service import shift_object_center_to_origin
        self.centered_ycb_point_normal_by_whole, self.t_shift_ycb_by_whole = shift_object_center_to_origin(
            target_point_normal = self.ycb_point_normal,
        )

    # def shift_ycb_whole_object_center_with_contact_info(self):
    #     from value_object import PointNormalUnitPairs
    #     self.centered_ycb_point_normal = PointNormalUnitPairs(
    #             points  = (self.centered_ycb_point_normal_by_whole.points + self.t_shift_ycb_contact_by_contact),
    #             normals = self.centered_ycb_point_normal_by_whole.normals,
    #     )
    #     self.object_whole_surface = self.centered_ycb_point_normal


    # def build_ycb_whole_surface_without_centering(self):
    #     from value_object import PointNormalUnitPairs
    #     # whole をそのまま「object frame の surface」として使う
    #     self.centered_ycb_point_normal = PointNormalUnitPairs(
    #         points  = self.ycb_point_normal.points,
    #         normals = self.ycb_point_normal.normals,
    #     )
    #     self.object_whole_surface = self.centered_ycb_point_normal
    #     # import ipdb; ipdb.set_trace()



    def build_ycb_whole_surface(self):
        from value_object import PointNormalUnitPairs
        # ★ object frame の whole surface をそのまま使う
        self.object_whole_surface = PointNormalUnitPairs(
            points  = self.ycb_point_normal.points,
            normals = self.ycb_point_normal.normals,
        )

    def build_manual_data_point_cloud(self):
        from value_object import PointNormalUnitPairs
        from point_cloud_generation._generate import _generate
        # -----
        points, normals = _generate()
        self.point_cloud = PointNormalUnitPairs(points, normals)


    def build_single_fingertip_target_box(self, j: int):
        from value_object import PointNormalCorrespondencePairs, PointNormalUnitPairs
        from service import generate_surface_points_panda_hand
        # -----
        base_source_points   = generate_surface_points_panda_hand(
            num_fingertip_surface_points = self.num_fingertip_surface_points,
            d0                           = 0.0,
            finger_index                 = j,
        )
        # -----
        object_half_size = self.config_pc_data.object_half_size
        # -----
        base_target_points  = np.copy(base_source_points)
        target_points       = base_target_points + (-1)**(j)*(self.v0*object_half_size)
        target_normals      = np.zeros_like(target_points) + (-1)**(j)*(self.v0)
        # -----
        return PointNormalUnitPairs(
            points  = target_points,
            normals = target_normals,
        )

    def build_paired_finger_box_target(self):
        from value_object import PointNormalUnitPairs
        from copy import deepcopy
        # ---
        right_target : PointNormalUnitPairs = self.build_single_fingertip_target_box(j=1)
        left_target  : PointNormalUnitPairs = self.build_single_fingertip_target_box(j=2)
        # ---
        self.target = PointNormalUnitPairs(
            points  = np.concatenate([right_target.points, left_target.points]),
            normals = np.concatenate([right_target.normals, left_target.normals]),
        )
        # ----
        self.contact_indices        = np.arange(self.target.points.shape[0])
        self.object_contact_surface = deepcopy(self.target)
        self.object_whole_surface   = deepcopy(self.target)

        self.model_name = "Box"

    def build_point_cloud_dataset(self):
        from value_object import PointNormalCorrespondencePairs
        self.point_cloud_dataset = PointNormalCorrespondencePairs(
            source = self.source,
            target = self.target,
        )

    def build_point_cloud_dataset_from_ycb_whole_data(self):
        from value_object import PointNormalCorrespondencePairs
        self.point_cloud_dataset = PointNormalCorrespondencePairs(
            source = self.source,
            target = self.ycb_point_normal,
        )


    def build_point_cloud_clustering(self, N_CLUSTERS: int = 8):
        from point_cloud_clustering.compute_cluster_centroids import compute_cluster_centroids
        # ----
        centers, labels = compute_cluster_centroids(
            points       = np.array(self.object_whole_surface.points),
            n_clusters   = N_CLUSTERS,
            random_state = 0,
        )
        self.centers = centers
        self.labels  = labels


    def build_fingertip_source(self, finger_index: int, normal_vector: np.ndarray):
        from value_object import PointNormalIndexUnitPairs
        from hand_surface_generation.generate_surface import generate_surface
        # ----- right ------
        source_points = generate_surface(
            distance_between_points = self.config_gripper_surface.distance_between_points,
            gripper_z_width         = self.config_gripper_surface.gripper_z_width,
            gripper_x_width         = self.config_gripper_surface.gripper_x_width,
            d0                      = self.config_gripper_surface.d0,
            finger_index            = finger_index,
            robot_name              = self.robot_name,
        )

        source_normals = (np.zeros_like(source_points) + normal_vector)
        # -----
        self.source = PointNormalIndexUnitPairs(
            points         = source_points,
            normals        = source_normals,
            finger_indices = np.zeros(source_points.shape[0]) + finger_index,
        )
        return deepcopy(self.source)


    def build_paired_finger_source(self):
        from value_object import PointNormalIndexUnitPairs
        # ---
        right_source : PointNormalIndexUnitPairs = self.build_fingertip_source(finger_index=1, normal_vector=np.array([0, 1, 0]))
        left_source  : PointNormalIndexUnitPairs = self.build_fingertip_source(finger_index=2, normal_vector=np.array([0, -1, 0]))
        # ---
        self.num_fingertip_surface_points = len(right_source.finger_indices)
        # ---
        self.source = PointNormalIndexUnitPairs(
            points         = np.concatenate([right_source.points, left_source.points]),
            normals        = np.concatenate([right_source.normals, left_source.normals]),
            finger_indices = np.concatenate([right_source.finger_indices, left_source.finger_indices]),
        )

    def build_hand_origin(self):
        from value_object import PointNormalUnitPairs
        self.hand_origin = PointNormalUnitPairs(
            points  = np.array(self.contact_plane_origin).reshape(1, -1),
            normals = np.array(self.contact_plane_normal).reshape(1, -1),
        )

    def build_single_finger_indices(self, j: int):
        from service import create_single_finger_indices

        # num_fingertip_surface_points =
        # import ipdb; ipdb.set_trace()
        self.finger_indices = create_single_finger_indices(
            num_fingertip_surface_points = self.num_fingertip_surface_points,
            j = j,
        )
        return deepcopy(self.finger_indices)

    def build_paired_finger_indices(self):
        # ----
        right_index = self.build_single_finger_indices(j=1)
        left_index  = self.build_single_finger_indices(j=2)
        # ----
        self.finger_indices = np.hstack([right_index, left_index])








    # --- SDF スコア用ヘルパー ---

    def _rho_band(self, d, band=0.005):
        d = np.asarray(d)
        return np.maximum(0.0, band - np.abs(d))

    def score_hand_pose_by_sdf(self, R_mat: np.ndarray, t: np.ndarray,
                               band: float = 0.005) -> float:
        """
        R_mat: (3,3) hand rotation (world frame)
        t    : (3,)  hand origin position (world frame)
        """
        F_local = self.source.points             # (M,3) canonical fingertip surface
        P_world = (R_mat @ F_local.T).T + t      # (M,3)

        d = self.pcd_sdf.sample_sdf_nearest(P_world)
        val = self._rho_band(d, band=band)
        return float(np.sum(val))


    def search_initial_pose_by_sdf(self,
                                   dx_range=(-0.02, 0.02),
                                   dy_range=(-0.02, 0.02),
                                   num_xy=7,
                                   yaw_range=(-np.pi/2, np.pi/2),
                                   num_yaw=9,
                                   band=0.005):
        """
        現在の hand 初期姿勢の周りで (x,y,yaw) をグリッドサーチし、
        SDF スコアが最大になる (R*, t*) を返す。
        """
        from scipy.spatial.transform import Rotation as R
        # 1) 現在の hand pose R0, t0 をどこかから取得
        R0 = R.from_rotvec(self.gripper_rotvec).as_matrix()
        t0 = self.gripper_translation          # (3,)

        dxs = np.linspace(dx_range[0], dx_range[1], num_xy)
        dys = np.linspace(dy_range[0], dy_range[1], num_xy)
        yaws = np.linspace(yaw_range[0], yaw_range[1], num_yaw)

        best_score = -np.inf
        best_R = None
        best_t = None

        for yaw in yaws:
            R_yaw = R.from_euler("z", yaw).as_matrix()
            R_cand = R_yaw @ R0

            for dx in dxs:
                for dy in dys:
                    t = t0 + np.array([dx, dy, 0.0])

                    score = self.score_hand_pose_by_sdf(R_cand, t, band=band)

                    if score > best_score:
                        best_score = score
                        best_R = R_cand.copy()
                        best_t = t.copy()

        print("[Stage-0] best_score:", best_score)
        print("[Stage-0] best_t:", best_t)

        self.best_R = best_R
        self.best_t = best_t

        # ==============================================
        from service import ExtendedRotation as ER
        print("Update gripper_rotvec! -> ", R.from_euler("z", yaw).as_matrix())
        self.gripper_rotvec = ER.from_matrix(best_R).as_rotvec()
        self.gripper_translation = best_t
        #  ==============================================

        # import ipdb; ipdb.set_trace()
        # return best_R, best_t


    def build_palm_least_square(self):
        from grasp_planning.least_square import PalmPoseLeastSquares
        self.palm_least_square = PalmPoseLeastSquares(self)

    def build_coupling_palm_least_square(self):
        from grasp_planning.least_square import CouplingPalmPoseLeastSquares
        self.coupling_palm_least_square = CouplingPalmPoseLeastSquares(self)

    def build_palm_approach_least_square(self):
        from grasp_planning.least_square import PalmPoseApproachRLestSquare
        self.palm_approach_least_square = PalmPoseApproachRLestSquare(self)


    def build_disf_palm_R_ls(self):
        from grasp_planning.disf.optimization.least_square import PalmRotationLeastSquare_EnEa
        self.disf_palm_R_ls = PalmRotationLeastSquare_EnEa(self)

    def build_disf_finger_ls_Ep(self):
        from grasp_planning.disf.optimization.least_square import Point2PlaneLeastSquareWithFingertipDisplacement
        self.disf_finger_ls_Ep = Point2PlaneLeastSquareWithFingertipDisplacement(self)

    def build_palm_point2plane_least_square(self):
        from grasp_planning.least_square import PalmPosePoint2PlaneLS
        self.palm_point2plane_least_square = PalmPosePoint2PlaneLS(self)

    def build_palm_normal_alignment_least_square(self):
        from grasp_planning.least_square import PalmPoseNormalAlignmentLS
        self.palm_normal_alignment_least_square = PalmPoseNormalAlignmentLS(self)


    def build_finger_least_square(self):
        from grasp_planning.least_square import FingertipDisplacementLeastSquare
        self.finger_least_square = FingertipDisplacementLeastSquare(self)

    def build_set_error(self):
        from grasp_planning.utils.error_computation import SetErrorComputation
        from grasp_planning.utils.error_computation import EpComputation
        from grasp_planning.utils.error_computation import EnComputation
        from grasp_planning.utils.error_computation import EaComputation
        # ----
        self.verbose_error = self.config_isf.verbose.textlog.error
        # ----
        self.Ep = EpComputation(self)
        self.En = EnComputation(self)
        self.Ea = EaComputation(self)
        # ---- E_total -----
        self.error = SetErrorComputation(self)
        # ----- E_CoM -----
        from grasp_planning.utils.error_computation import GeomError
        self.geom_error = GeomError(self)
        # ----- E_CoM -----
        from grasp_planning.utils.error_computation import CoMError
        self.com_error = CoMError(self)

    '''
        VISF: Vanilla Iterative Surface Fitting
    '''
    def build_visf_palm_Rt_opt(self):
        from grasp_planning.visf.optimization import VISF_PalmOptimization
        self.visf_palm_Rt_opt = VISF_PalmOptimization(self)

    def build_visf_finger_opt(self):
        from grasp_planning.visf.optimization import IPFO_FingertipDisplacementOptimization
        self.visf_finger_opt = IPFO_FingertipDisplacementOptimization(self)

    def build_visf_palm_Rt_EpEnEa_least_square(self):
        from grasp_planning.visf.optimization.least_square import PalmLeastSquare_EpEnEa
        self.visf_palm_Rt_ls = PalmLeastSquare_EpEnEa(self)

    def build_visf_fingertip_displacement_least_square(self):
        from grasp_planning.visf.optimization.least_square import FingertipDisplacement
        self.visf_finger_ls = FingertipDisplacement(self)

    '''
        DISF: Disentangled Iterative Surface Fitting
    '''
    def build_disf_palm_R_opt(self):
        from grasp_planning.disf.optimization import DISF_PalmRotationOptimization
        self.disf_palm_R_opt = DISF_PalmRotationOptimization(self)

    def build_disf_trans_centroid(self):
        from grasp_planning.disf.optimization import DISF_TranslationCentroid
        self.disf_trans_centroid = DISF_TranslationCentroid(self)

    def build_disf_finger_opt(self):
        from grasp_planning.disf.optimization import DISF_FingertipDisplacementOptimization
        self.disf_finger_opt = DISF_FingertipDisplacementOptimization(self)


    def build_isf_visualizer(self):
        from grasp_planning.disf.visualization import DISFVisualization
        self.isf_visualizer = DISFVisualization(self)

    def build_ipfo_error_computation(self):
        from grasp_planning.visf import IPFO_ErrorCompute
        self.ipfo_error_computation = IPFO_ErrorCompute(self)

    def build_ipfo_alpha_vs_rotation_est(self):
        from grasp_planning.visf import IPFO_alpha_vs_rotation_est
        self.ipfo_alpha_vs_rotation_est = IPFO_alpha_vs_rotation_est(self)

    def build_vanilla_isf(self):
        from grasp_planning.visf import VISF
        self.visf = VISF(self)
        self.isf  = self.visf

    def build_disf(self):
        from grasp_planning.disf import DISF
        self.disf = DISF(self)
        self.isf  = self.disf

    def build_cmasf(self):
        from grasp_planning.cma import CMASF
        self.cmasf = CMASF(self)
        self.isf   = self.cmasf

    def build_isf_error_compute_static_pair(self):
        from grasp_planning.visf import ISF_ErrorCompute_StaticPair
        self.isf_error_compute_static_pair = ISF_ErrorCompute_StaticPair(self)

    def build_isf_alpha_vs_rotation_est_static_pair(self):
        from grasp_planning.visf import ISF_alpha_vs_rotation_est_StaticPair
        self.isf_alpha_vs_rotation_est_static_pair = ISF_alpha_vs_rotation_est_StaticPair(self)

    def build_isf_loop_with_single_icp(self):
        from grasp_planning.disf import ISF_Planning_With_Single_ICP
        self.isf_planning = ISF_Planning_With_Single_ICP(self)

    def build_icp_matcher(self):
        from correspondence_matching import ICPPointMatcherWithNormals
        self.icp_matcher = ICPPointMatcherWithNormals(self)

    def save_configs(self):
        import hydra
        from omegaconf import OmegaConf
        from hydra import utils
        # ----
        config_names = [
            "config_ipfo",
            "config_env",
            "config_ik_solver",
            "config_grasp_evaluation",
            "config_pc_data",
            "config_icp",
        ]
        # ------
        # import ipdb; ipdb.set_trace()
        for config_name in config_names:
            if hasattr(self, config_name):  # 属性が存在する場合のみ処理
                config = getattr(self, config_name)  # 動的に属性を取得
                OmegaConf.save(config=config, f=os.path.join(self.results_save_dir, f"{config_name}.yaml"))
