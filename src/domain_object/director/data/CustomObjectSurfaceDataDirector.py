from domain_object.builder import SelfContainedDomainObjectBuilder


class CustomObjectSurfaceDataDirector:
    @staticmethod
    def construct(
            builder    : SelfContainedDomainObjectBuilder,
            robot_name : str,
            object_name: str,
            xml_scene_file_generation: bool = True,
        ):
        # ---
        builder.build_config_point_cloud_data(config_name=object_name)

        if xml_scene_file_generation:
            builder.build_custom_xml_scene_file(robot_name, object_name)

        builder.build_custom_stl_data_point_cloud()
        builder.build_ycb_data_normal_estimation()
        builder.build_ycb_data_PointNormal_ValueObject()
        builder.build_ycb_target_by_contact_estimation_with_gripper_box()
        builder.build_ycb_whole_surface()
        # ---
        return builder.get_domain_object()

