from domain_object.builder import SelfContainedDomainObjectBuilder


class PFOCommonDirector:
    @staticmethod
    def construct(
            builder            : SelfContainedDomainObjectBuilder,
            robot_name         : str,
            object_name        : str,
            isf_model          : str,
        ):
        # ---------
        builder.build_config_gripper_surface(config_name=robot_name)
        builder.build_config_isf(config_name=isf_model)
        builder.build_ipfo_parameters()
        builder.build_ipfo_text_logger()
        builder.build_config_point_cloud_data(config_name=object_name)
        builder.build_loop_criteria()
        # ---------
        return builder.get_domain_object()

