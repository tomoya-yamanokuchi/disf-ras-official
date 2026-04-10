from pathlib import Path
import numpy as np

from mujoco_grasping import ISFMujocoGraspingWithPosePerturbation
from args import parse_args


def build_save_path(
    results_save_dir : str | Path,
    total_mass       : float,
):
    results_save_dir = Path(results_save_dir)
    results_save_dir.mkdir(parents = True, exist_ok = True)
    return results_save_dir / f"all_results_total_mass_{total_mass:.2f}.npy"


def run(
    robot_name       ,
    object_name_list ,
    isf_model        ,
    n_trials         ,
    seed             ,
    max_angle_deg    ,
    total_mass       ,
    # results_save_dir ,
):
    grasp = ISFMujocoGraspingWithPosePerturbation()
    all_results = grasp.evaluate(
        robot_name       = robot_name,
        object_name_list = object_name_list,
        isf_model        = isf_model,
        n_trials         = n_trials,
        seed             = seed,
        max_angle_deg    = max_angle_deg,
    )

    saved_data = {
        "total_mass" : total_mass,
        "results"    : list(all_results.values()),
    }

    save_path = build_save_path(
        results_save_dir = results_save_dir,
        total_mass       = total_mass,
    )
    np.save(
        save_path,
        saved_data,
        allow_pickle = True,
    )

    print(f"[Saved] {save_path}")
    return saved_data


if __name__ == "__main__":
    args = parse_args()
    run(
        robot_name       = args.robot_name,
        object_name_list = args.object_name.split(","),
        isf_model        = args.method,
        n_trials         = args.n_trials,
        seed             = args.seed,
        max_angle_deg    = args.max_angle_deg,
        total_mass       = args.total_mass,
        # results_save_dir = args.results_save_dir,
    )
