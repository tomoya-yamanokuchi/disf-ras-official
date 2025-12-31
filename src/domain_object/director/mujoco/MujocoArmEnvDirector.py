from domain_object.builder import SelfContainedDomainObjectBuilder

class MujocoArmEnvDirector:
    @staticmethod
    def construct(
            builder        : SelfContainedDomainObjectBuilder,
            robot_name     : str = "panda",
            # ---
            robot_mode     : str = "grasp",
            load_xml_path  : str = None,
            # ---
        ) -> SelfContainedDomainObjectBuilder:
        # ---
        builder.build_config_env(config_name=robot_name + f"_{robot_mode}")
        builder.build_config_ik_solver()
        # ----
        builder.build_model(load_xml_path)
        builder.buid_data()
        builder.buid_body()
        builder.buid_site()
        builder.build_geom()
        builder.build_camera()
        builder.build_option()
        builder.build_scene()
        builder.build_viewer()
        # ---
        builder.build_ik_solver()
        # ----
        builder.build_env()
        # ---
        builder.build_config_grasp_evaluation()
        builder.build_grasp_evaluator()
        # ---
        return builder.get_domain_object()
