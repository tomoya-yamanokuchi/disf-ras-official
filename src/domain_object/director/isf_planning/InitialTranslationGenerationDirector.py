from domain_object.builder import SelfContainedDomainObjectBuilder
from ..isf import DISF_Director
from ..isf import VISF_Director
from ..isf import CMA_Director
from ..icp import ICP
from ..isf import PFOCommonDirector
from ..data import GripperSurfaceDataDirector, YCBObjectSurfaceDataDirector, RectPlaneSurfaceDataDirector, ObservedObjectSurfaceDataDirector
from ..data import CustomObjectSurfaceDataDirector


class InitialTranslationGenerationDirector:
    @staticmethod
    def construct(
            builder    : SelfContainedDomainObjectBuilder,
            robot_name : str,
            object_name: str,
            isf_model  : str,
            N_CLUSTERS : int,
        ):


        builder.set_current_working_dir()
        builder.set_robot_name(robot_name)
        builder.set_object_name(object_name)
        # ---
        # import ipdb; ipdb.set_trace()
        # builder.build_config_env(config_name=robot_name + f"_{robot_name}")

        # ==================== common setting  ====================
        if isf_model in ["cma", "visf", "disf"]:
            builder = PFOCommonDirector.construct(
                builder, robot_name, object_name, isf_model)
        else:
            raise NotImplementedError()

        # =============== Object Surface Generation ===============
        if object_name == "box":
            import numpy as np
            builder = RectPlaneSurfaceDataDirector.construct(builder, robot_name, object_name)
            # import ipdb; ipdb.set_trace()
        elif "custom" in object_name:
            builder = CustomObjectSurfaceDataDirector.construct(builder, robot_name, object_name, xml_scene_file_generation=False)
        elif "observed" in object_name:
            builder = ObservedObjectSurfaceDataDirector.construct(builder, robot_name, object_name)
        else:
            builder = YCBObjectSurfaceDataDirector.construct(builder, robot_name, object_name, xml_scene_file_generation=False)


        builder.build_point_cloud_clustering(N_CLUSTERS)

        # # ---
        return builder.get_domain_object()

