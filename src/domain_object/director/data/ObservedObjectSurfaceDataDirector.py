from domain_object.builder import SelfContainedDomainObjectBuilder


class ObservedObjectSurfaceDataDirector:
    @staticmethod
    def construct(
            builder    : SelfContainedDomainObjectBuilder,
            robot_name : str,
            object_name: str,
            xml_scene_file_generation: bool = True,
        ):
        # ---
        builder.build_config_point_cloud_data(config_name=object_name)

        # if xml_scene_file_generation:
        #     builder.build_ycb_xml_scene_file(robot_name, object_name)



        builder.build_observed_pcd()
        builder.build_ycb_data_normal_estimation()

        # builder.visualize_point_cloud_with_normals(
        #     normal_length=0.01,   # オブジェクトのスケールに合わせて調整
        #     max_num_normals=2000, # 多いようなら減らす
        # )

        builder.build_ycb_data_PointNormal_ValueObject()
        builder.build_ycb_target_by_contact_estimation_with_gripper_box()

        builder.build_ycb_whole_surface()
        # ---
        return builder.get_domain_object()

