from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen = True)
class BoxComposite5CoMResult:
    total_mass           : float
    slice_centers_x      : tuple[float, float, float, float, float]
    slice_masses         : tuple[float, float, float, float, float]
    true_com_x           : float
    geometric_centroid_x : float
    com_shift_x          : float


@dataclass(frozen = True)
class BoxComposite5MomentArmResult:
    grasp_x                   : float
    moment_arm_x              : float
    abs_moment_arm_x          : float
    normalized_moment_arm_x   : float


def compute_box_composite5_true_com(
    full_size_xyz : tuple[float, float, float],
    slice_masses  : tuple[float, float, float, float, float],
) -> BoxComposite5CoMResult:
    if len(full_size_xyz) != 3:
        raise ValueError("full_size_xyz must have exactly 3 elements.")
    if len(slice_masses) != 5:
        raise ValueError("slice_masses must have exactly 5 elements.")

    lx, ly, lz = full_size_xyz
    if lx <= 0 or ly <= 0 or lz <= 0:
        raise ValueError(f"All box dimensions must be positive, got {full_size_xyz}.")
    if any(m <= 0 for m in slice_masses):
        raise ValueError(f"All slice masses must be positive, got {slice_masses}.")

    slice_width     = lx / 5.0
    slice_centers_x = tuple(
        -lx / 2.0 + slice_width / 2.0 + i * slice_width
        for i in range(5)
    )

    total_mass           = sum(slice_masses)
    true_com_x           = sum(m * x for m, x in zip(slice_masses, slice_centers_x)) / total_mass
    geometric_centroid_x = 0.0
    com_shift_x          = true_com_x - geometric_centroid_x

    return BoxComposite5CoMResult(
        total_mass           = total_mass,
        slice_centers_x      = slice_centers_x,
        slice_masses         = slice_masses,
        true_com_x           = true_com_x,
        geometric_centroid_x = geometric_centroid_x,
        com_shift_x          = com_shift_x,
    )


def compute_normalized_com_shift(
    full_size_xyz  : tuple[float, float, float],
    com_shift_x    : float,
    normalization  : str = "length_x",
) -> float:
    lx, ly, lz = full_size_xyz

    if normalization == "length_x":
        denom = lx
    elif normalization == "half_length_x":
        denom = lx / 2.0
    elif normalization == "bbox_diagonal":
        denom = (lx**2 + ly**2 + lz**2) ** 0.5
    else:
        raise ValueError(f"Unknown normalization: {normalization}")

    if denom <= 0:
        raise ValueError("Normalization denominator must be positive.")

    return com_shift_x / denom


def compute_box_composite5_normalized_moment_arm(
    full_size_xyz : tuple[float, float, float],
    true_com_x    : float,
    grasp_x       : float,
) -> BoxComposite5MomentArmResult:
    if len(full_size_xyz) != 3:
        raise ValueError("full_size_xyz must have exactly 3 elements.")

    lx, ly, lz = full_size_xyz
    if lx <= 0 or ly <= 0 or lz <= 0:
        raise ValueError(f"All box dimensions must be positive, got {full_size_xyz}.")

    moment_arm_x            = true_com_x - grasp_x
    abs_moment_arm_x        = abs(moment_arm_x)
    normalized_moment_arm_x = abs_moment_arm_x / lx

    return BoxComposite5MomentArmResult(
        grasp_x                 = grasp_x,
        moment_arm_x            = moment_arm_x,
        abs_moment_arm_x        = abs_moment_arm_x,
        normalized_moment_arm_x = normalized_moment_arm_x,
    )
