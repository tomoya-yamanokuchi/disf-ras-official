
from domain_object.director.mujoco import MujocoGrasping_with_ISF_Planning
from domain_object.builder import SelfContainedDomainObjectBuilder, DomainObject
from .MujocoGraspingArmGraspViaPregrasp import MujocoGraspingArmGraspViaPregrasp
from typing import List


class ISFMujocoGraspingYCBObjectEvaluation:
    def __init__(self):
        pass

    def evaluate(self,
            robot_name      : str       = "panda",
            object_name_list: List[str] = ["006_mustard_bottle"],
            isf_model       : str       = "dummy",
        ):
        print(f"-------------------------------")
        for object_name in object_name_list:
            print(f"object_name = {object_name}")
            # --------------------------------

            # --------------------------------
            builder       = SelfContainedDomainObjectBuilder()
            director      = MujocoGrasping_with_ISF_Planning()
            domain_object = director.construct(
                builder         = builder,
                robot_name      = robot_name,
                object_name     = object_name,
                isf_model       = isf_model,
            )
            # --------------------------------
            isf_planning = domain_object.isf_planning
            isf_results  = isf_planning.run(return_all=True)
            # import ipdb; ipdb.set_trace()
            # --------------------------------
            grasp = MujocoGraspingArmGraspViaPregrasp(domain_object)
            grasp.execute(isf_results)
        print(f"-------------------------------")
