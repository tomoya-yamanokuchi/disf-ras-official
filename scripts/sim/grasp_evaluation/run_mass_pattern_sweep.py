from __future__ import annotations
import sys
from pathlib import Path
from args import parse_args


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from run_one_mass_pattern import (
    run as run_one_mass_pattern,
)



DEFAULT_CONDITIONS = (
    "large_left",
    # "medium_left",
    # "mild_left",
    # "uniform",
    # "mild_right",
    # "medium_right",
    "large_right",
)

#  "large_left",
# "large_right",

def parse_total_mass_list(
    total_mass_list_str: str,
) -> list[float]:
    return [
        float(x.strip())
        for x in total_mass_list_str.split(",")
        if x.strip()
    ]


def parse_condition_list(
    condition_list_str: str | None,
) -> list[str]:
    if condition_list_str is None:
        return list(DEFAULT_CONDITIONS)

    parsed = [
        x.strip()
        for x in condition_list_str.split(",")
        if x.strip()
    ]
    if len(parsed) == 0:
        return list(DEFAULT_CONDITIONS)

    return parsed


def run(
    robot_name        ,
    isf_model         ,
    total_mass_list   ,
    condition_list    ,
    bias_mild         ,
    bias_medium       ,
    bias_large        ,
    grasp_x           ,
    size_x            ,
    size_y            ,
    size_z            ,
    n_trials          ,
    seed              ,
    max_angle_deg     ,
    object_root_dir   ,
    results_save_dir  ,
):
    all_saved_data = []

    for total_mass in total_mass_list:
        for condition in condition_list:
            saved_data = run_one_mass_pattern(
                robot_name       = robot_name,
                isf_model        = isf_model,
                condition        = condition,
                total_mass       = total_mass,
                bias_mild        = bias_mild,
                bias_medium      = bias_medium,
                bias_large       = bias_large,
                grasp_x          = grasp_x,
                size_x           = size_x,
                size_y           = size_y,
                size_z           = size_z,
                n_trials         = n_trials,
                seed             = seed,
                max_angle_deg    = max_angle_deg,
                object_root_dir  = object_root_dir,
                results_save_dir = results_save_dir,
            )
            all_saved_data.append(saved_data)

    return all_saved_data


if __name__ == "__main__":
    args = parse_args()

    total_mass_list = parse_total_mass_list(
        total_mass_list_str = args.total_mass_list,
    )
    condition_list = parse_condition_list(
        condition_list_str = getattr(args, "condition_list", None),
    )


    run(
        robot_name       = args.robot_name,
        isf_model        = args.method,
        total_mass_list  = total_mass_list,
        condition_list   = condition_list,
        bias_mild        = args.bias_mild,
        bias_medium      = args.bias_medium,
        bias_large       = args.bias_large,
        grasp_x          = args.grasp_x,
        size_x           = args.size_x,
        size_y           = args.size_y,
        size_z           = args.size_z,
        n_trials         = args.n_trials,
        seed             = args.seed,
        max_angle_deg    = args.max_angle_deg,
        object_root_dir  = args.object_root_dir,
        results_save_dir = args.results_save_dir,
    )

'''
    python scripts/sim/grasp_evaluation/run_mass_pattern_sweep.py --robot_name panda --method disf --total_mass_list "0.60"
'''

