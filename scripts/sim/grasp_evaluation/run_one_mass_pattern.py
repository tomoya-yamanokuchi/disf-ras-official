from __future__ import annotations

from pathlib import Path

from args import parse_args
from box_composite5.pattern_runner import BoxComposite5PatternRunner
from box_composite5.spec import BoxComposite5PatternSpec


def run(
    robot_name       ,
    isf_model        ,
    condition        ,
    total_mass       ,
    bias_mild        ,
    bias_medium      ,
    bias_large       ,
    grasp_x          ,
    size_x           ,
    size_y           ,
    size_z           ,
    n_trials         ,
    seed             ,
    max_angle_deg    ,
    object_root_dir  ,
    results_save_dir ,
):
    spec = BoxComposite5PatternSpec(
        robot_name       = robot_name,
        isf_model        = isf_model,
        condition        = condition,
        total_mass       = total_mass,
        bias_mild        = bias_mild,
        bias_medium      = bias_medium,
        bias_large       = bias_large,
        grasp_x          = grasp_x,
        full_size_xyz    = (size_x, size_y, size_z),
        n_trials         = n_trials,
        seed             = seed,
        max_angle_deg    = max_angle_deg,
        object_root_dir  = Path(object_root_dir),
        results_save_dir = Path(results_save_dir),
    )

    runner = BoxComposite5PatternRunner(
        spec = spec,
    )
    return runner.run()


if __name__ == "__main__":
    args = parse_args()

    run(
        robot_name       = args.robot_name,
        isf_model        = args.method,
        condition        = args.condition,
        total_mass       = args.total_mass,
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
