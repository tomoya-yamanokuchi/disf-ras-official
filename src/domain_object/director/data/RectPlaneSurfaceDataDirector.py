from domain_object.builder import SelfContainedDomainObjectBuilder


class RectPlaneSurfaceDataDirector:
    """
    矩形平面の surface 点群を使った grasp planning 用の
    SelfContainedDomainObjectBuilder の組み立て手順を定義するディレクタ。

    YCBObjectSurfaceDataDirector と同様に、
    - config_pc_data の読み込み
    - 点群の生成
    - normal 推定と PointNormal 化
    - contact 平面との交差領域の抽出
    - center shift と dataset 構築
    をまとめて行う。
    """

    @staticmethod
    def construct(
        builder: SelfContainedDomainObjectBuilder,
        robot_name : str,
        object_name: str,
    ):
        # --- config 読み込み（contact 平面や gripper_pose などをセット） ---
        builder.build_config_point_cloud_data(config_name=object_name)
        builder.build_ycb_xml_scene_file(robot_name, object_name)

        # --- 矩形平面の surface 点群を生成 ---
        # ここで SelfContainedDomainObjectBuilder.point_cloud が open3d.geometry.PointCloud としてセットされる想定
        builder.build_manual_data_point_cloud() # normal 生成も含む
        builder.build_ycb_data_PointNormal_ValueObject()
        builder.build_ycb_target_by_contact_estimation_with_gripper_box()

        builder.build_ycb_whole_surface()
        # ---
        return builder.get_domain_object()
