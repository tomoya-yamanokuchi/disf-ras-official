from __future__ import annotations

from pathlib import Path
import numpy as np

from .object_set_generator import BoxCompositeObjectSet


def build_save_path(
    results_save_dir : Path,
    total_mass       : float,
) -> Path:
    results_save_dir.mkdir(parents = True, exist_ok = True)
    filename = f"all_results_total_mass_{total_mass:.2f}.npy"
    return results_save_dir / filename


def save_mass_sweep_results(
    save_path         : Path,
    object_set        : BoxCompositeObjectSet,
    condition_results : dict,
):
    saved_data = {
        "experiment_metadata": {
            "robot_name"      : object_set.spec.robot_name,
            "isf_model"       : object_set.spec.isf_model,
            "total_mass"      : object_set.spec.total_mass,
            "bias_levels"     : {
                "mild"   : object_set.spec.bias_levels.mild,
                "medium" : object_set.spec.bias_levels.medium,
                "large"  : object_set.spec.bias_levels.large,
            },
            "full_size_xyz"   : object_set.spec.full_size_xyz,
            "grasp_x"         : object_set.spec.grasp_x,
            "n_trials"        : object_set.spec.n_trials,
            "seed"            : object_set.spec.seed,
            "max_angle_deg"   : object_set.spec.max_angle_deg,
            "object_root_dir" : str(object_set.spec.object_root_dir),
            "results_save_dir": str(object_set.spec.results_save_dir),
        },
        "condition_results": condition_results,
    }

    np.save(
        save_path,
        saved_data,
        allow_pickle = True,
    )
