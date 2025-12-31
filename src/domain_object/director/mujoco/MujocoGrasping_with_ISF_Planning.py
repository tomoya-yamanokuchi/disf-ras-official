from domain_object.builder import SelfContainedDomainObjectBuilder
from .MujocoArmEnvDirector import MujocoArmEnvDirector
from ..isf_planning import ISF_Planning_With_YCB_Object_Director

class MujocoGrasping_with_ISF_Planning:
    @staticmethod
    def construct(
            builder        : SelfContainedDomainObjectBuilder,
            robot_name     : str = "panda",
            object_name    : str = None,
            isf_model      : str = None,
            robot_mode     : str = "grasp",
            load_xml_path  : str = None,
        ):
        builder.set_current_working_dir()
        builder.set_robot_name(robot_name)
        builder.set_object_name(object_name)
        # ---
        builder.build_config_env(config_name=robot_name + f"_{robot_mode}")
        builder = ISF_Planning_With_YCB_Object_Director.construct(builder, robot_name, object_name, isf_model)
        builder = MujocoArmEnvDirector.construct(builder, robot_name, robot_mode, load_xml_path)
        builder.save_configs()
        # ---
        return builder.get_domain_object()

