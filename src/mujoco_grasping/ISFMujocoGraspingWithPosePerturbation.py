from __future__ import annotations

import numpy as np

from domain_object.builder import SelfContainedDomainObjectBuilder
from pose_perturbation.rotation_noise import RotationPerturbationConfig
from .MujocoGraspingWithPosePerturbation import (
    MujocoGraspingWithPosePerturbation,
)

from domain_object.director.mujoco.MujocoGrasping_with_ISF_Planning import (
    MujocoGrasping_with_ISF_Planning,
)


class ISFMujocoGraspingWithPosePerturbation:
    def evaluate(
        self,
        robot_name      : str,
        object_name_list: list[str],
        isf_model       : str,
        # -----
        # condition       : str,
        n_trials        : int   = 10,
        seed            : int   = 0,
        max_angle_deg   : float = 3.0,
    ):
        all_results = {}

        rotation_perturbation_config = RotationPerturbationConfig(
            max_angle_deg = max_angle_deg,
            euler_order   = "xyz",
        )
        rng = np.random.default_rng(seed)

        for object_name in object_name_list:
            print(f"object_name = {object_name}")
            # --------------------------------------------------
            builder       = SelfContainedDomainObjectBuilder()
            director      = MujocoGrasping_with_ISF_Planning()
            domain_object = director.construct(
                builder         = builder,
                robot_name      = robot_name,
                object_name     = object_name,
                isf_model       = isf_model,
            )
            # --------------------------------------------------
            isf_planning = domain_object.isf_planning
            isf_results  = isf_planning.run(return_all=True)

            # ============== grasp execution with pose perturbation ==============
            executor = MujocoGraspingWithPosePerturbation(
                domain_object
            )

            trial_results = []
            n_success = 0

            for trial_id in range(n_trials):
                result = executor.execute(
                    isf_result                   = isf_results,
                    rotation_perturbation_config = rotation_perturbation_config,
                    rng                          = rng,
                )
                result["trial_id"] = trial_id
                trial_results.append(result)
                n_success += int(result["success"])

            success_rate = n_success / n_trials


            signed_normalized_moment_arm_x = getattr(
                domain_object,
                "signed_normalized_moment_arm_x",
                None,
            )

            all_results[object_name] = {
                # "condition"                      : condition,
                "success_rate"                   : success_rate,
                "n_success"                      : n_success,
                "n_trials"                       : n_trials,
                "signed_normalized_moment_arm_x" : signed_normalized_moment_arm_x,
            }

        return all_results
