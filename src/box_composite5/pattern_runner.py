from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import numpy as np

from mujoco_grasping import ISFMujocoGraspingWithPosePerturbation

from .asset_builder import (
    BoxComposite5PatternAssetBuilder,
)
from .spec import BoxComposite5PatternSpec


class BoxComposite5PatternRunner:
    def __init__(
        self,
        spec : BoxComposite5PatternSpec,
    ) -> None:
        self.spec = spec

    def _build_save_path(self) -> Path:
        self.spec.results_save_dir.mkdir(parents = True, exist_ok = True)
        filename = f"mass_{self.spec.total_mass:.2f}__{self.spec.condition}.npy"
        return self.spec.results_save_dir / filename

    def _evaluate(
        self,
        object_name : str,
    ) -> dict:
        grasp = ISFMujocoGraspingWithPosePerturbation()
        all_results = grasp.evaluate(
            robot_name       = self.spec.robot_name,
            object_name_list = [object_name],
            isf_model        = self.spec.isf_model,
            n_trials         = self.spec.n_trials,
            seed             = self.spec.seed,
            max_angle_deg    = self.spec.max_angle_deg,
        )

        if object_name not in all_results:
            raise KeyError(f"{object_name} was not found in evaluation results.")

        return all_results[object_name]

    def run(self) -> dict:
        asset = BoxComposite5PatternAssetBuilder(
            spec = self.spec,
        ).build()

        eval_result = self._evaluate(
            object_name = asset.object_name,
        )

        saved_data = {
            "robot_name"                     : self.spec.robot_name,
            "isf_model"                      : self.spec.isf_model,
            "condition"                      : self.spec.condition,
            "object_name"                    : asset.object_name,
            "total_mass"                     : self.spec.total_mass,
            "slice_masses"                   : asset.slice_masses,
            "grasp_x"                        : self.spec.grasp_x,
            "full_size_xyz"                  : self.spec.full_size_xyz,
            "n_trials"                       : self.spec.n_trials,
            "seed"                           : self.spec.seed,
            "max_angle_deg"                  : self.spec.max_angle_deg,
            "success_rate"                   : eval_result["success_rate"],
            "n_success"                      : eval_result["n_success"],
            "signed_normalized_moment_arm_x" : asset.signed_normalized_moment_arm_x,
        }

        save_path = self._build_save_path()
        np.save(
            save_path,
            saved_data,
            allow_pickle = True,
        )

        print(f"[Saved] {save_path}")
        return saved_data
