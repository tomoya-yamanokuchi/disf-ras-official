from domain_object.builder import SelfContainedDomainObjectBuilder
from ..isf import DISF_Director
from ..isf import VISF_Director
from ..isf import CMA_Director
from ..icp import ICP
from ..isf import PFOCommonDirector
from ..data import GripperSurfaceDataDirector, YCBObjectSurfaceDataDirector, RectPlaneSurfaceDataDirector, ObservedObjectSurfaceDataDirector
from ..data import CustomObjectSurfaceDataDirector

class ISF_Planning_With_YCB_Object_Director:
    @staticmethod
    def construct(
            builder    : SelfContainedDomainObjectBuilder,
            robot_name : str,
            object_name: str,
            isf_model  : str,
        ):
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
            builder = CustomObjectSurfaceDataDirector.construct(builder, robot_name, object_name)
        elif "observed" in object_name:
            builder = ObservedObjectSurfaceDataDirector.construct(builder, robot_name, object_name)
        else:
            builder = YCBObjectSurfaceDataDirector.construct(builder, robot_name, object_name)

        # ============== Gripper Surface Generation  ==============
        builder = GripperSurfaceDataDirector.construct(builder)

        # ==================== ICP  ====================
        builder.build_point_cloud_dataset()
        builder = ICP.construct(builder)
        # =========== datatset generation ===========
        if   isf_model == "disf":
            builder = DISF_Director.construct(builder, config_name=object_name) # assumed to be same
        elif isf_model == "visf":
            builder = VISF_Director.construct(builder, config_name=object_name) # assumed to be same
        elif isf_model == "cma":
            builder = CMA_Director.construct(builder, config_name=object_name) # assumed to be same
        else:
            raise NotImplementedError()
        # ========== Sim evaluation metrics ==========
        # =============== ISF object ===============
        builder.build_isf_loop_with_single_icp()
        # ---
        return builder.get_domain_object()

