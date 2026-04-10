from __future__ import annotations

from box_composite_mass_sweep.object_set_generator import (
    BoxCompositeObjectSetGenerator,
)
from box_composite_mass_sweep.save_utils import (
    build_save_path,
    save_mass_sweep_results,
)
from box_composite5.spec import (
    BoxCompositeMassSweepSpec,
)

from .ISFMujocoGraspingWithPosePerturbation import (
    ISFMujocoGraspingWithPosePerturbation,
)


class ISFMujocoGraspingMassSweepEvaluation:
    def run(
        self,
        spec: BoxCompositeMassSweepSpec,
    ):
        # --------------------------------------------------
        # (1) generate STL / OBJ / XML and compute metadata
        # --------------------------------------------------
        object_set = BoxCompositeObjectSetGenerator(
            spec = spec,
        ).generate()

        # --------------------------------------------------
        # (2) run pose-perturbation grasp evaluation
        # --------------------------------------------------
        evaluator = ISFMujocoGraspingWithPosePerturbation()
        raw_saved_data = evaluator.evaluate(
            robot_name       = spec.robot_name,
            object_name_list = object_set.object_name_list,
            isf_model        = spec.isf_model,
            n_trials         = spec.n_trials,
            seed             = spec.seed,
            max_angle_deg    = spec.max_angle_deg,
            # total_mass       = spec.total_mass,
            # bias_levels      = {
            #     "mild"   : spec.bias_levels.mild,
            #     "medium" : spec.bias_levels.medium,
            #     "large"  : spec.bias_levels.large,
            # },
            # grasp_x          = spec.grasp_x,
            # full_size_xyz    = spec.full_size_xyz,
        )

        # --------------------------------------------------
        # (3) merge generated object metadata into results
        # --------------------------------------------------
        condition_results = raw_saved_data["condition_results"]

        for object_name, asset in object_set.condition_assets.items():
            if object_name not in condition_results:
                continue

            if "metadata" not in condition_results[object_name]:
                condition_results[object_name]["metadata"] = {}

            condition_results[object_name]["metadata"].update(
                {
                    "condition"                      : asset.condition,
                    "slice_masses"                   : asset.slice_masses,
                    "true_com_x"                     : asset.true_com_x,
                    "com_shift_x"                    : asset.com_shift_x,
                    "normalized_shift_x"             : asset.normalized_shift_x,
                    "grasp_x"                        : asset.grasp_x,
                    "moment_arm_x"                   : asset.moment_arm_x,
                    "normalized_moment_arm_x"        : asset.normalized_moment_arm_x,
                    "signed_normalized_moment_arm_x" : asset.signed_normalized_moment_arm_x,
                    "stl_path"                       : str(asset.stl_path),
                    "obj_path"                       : str(asset.obj_path),
                    "xml_path"                       : str(asset.xml_path),
                }
            )

        # --------------------------------------------------
        # (4) save
        # --------------------------------------------------
        save_path = build_save_path(
            results_save_dir = spec.results_save_dir,
            total_mass       = spec.total_mass,
        )
        save_mass_sweep_results(
            save_path         = save_path,
            object_set        = object_set,
            condition_results = condition_results,
        )

        return {
            "save_path"         : save_path,
            "saved_data"        : {
                "experiment_metadata": raw_saved_data["experiment_metadata"],
                "condition_results"  : condition_results,
            },
            "object_set"        : object_set,
        }
