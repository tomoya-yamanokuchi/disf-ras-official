from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen = True)
class BoxComposite5PatternSpec:
    robot_name       : str
    isf_model        : str

    condition        : str
    total_mass       : float

    bias_mild        : float
    bias_medium      : float
    bias_large       : float

    grasp_x          : float
    full_size_xyz    : tuple[float, float, float]

    n_trials         : int
    seed             : int
    max_angle_deg    : float

    object_root_dir  : Path
    results_save_dir : Path
