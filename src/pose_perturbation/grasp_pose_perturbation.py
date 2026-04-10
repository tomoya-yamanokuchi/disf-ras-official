from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rotation_noise import (
    RotationPerturbationConfig,
    apply_rotation_noise_to_quaternion_wxyz,
)


@dataclass(frozen=True)
class PerturbedGraspPoseResult:
    trial_id       : int
    quat_wxyz      : np.ndarray
    euler_noise_deg: np.ndarray


def generate_perturbed_grasp_quaternions(
    base_quat_wxyz : np.ndarray,
    n_trials       : int,
    rotation_config: RotationPerturbationConfig,
    seed           : int = 0,
) -> list[PerturbedGraspPoseResult]:
    """
    Generate multiple perturbed grasp orientations from one nominal quaternion.

    Translation is kept fixed. Only orientation is perturbed.
    """
    if n_trials <= 0:
        raise ValueError(f"n_trials must be positive, got {n_trials}.")

    rng = np.random.default_rng(seed)
    results: list[PerturbedGraspPoseResult] = []

    for trial_id in range(n_trials):
        quat_perturbed_wxyz, euler_noise_deg = apply_rotation_noise_to_quaternion_wxyz(
            base_quat_wxyz = base_quat_wxyz,
            config         = rotation_config,
            rng            = rng,
            left_multiply  = True,
        )
        results.append(
            PerturbedGraspPoseResult(
                trial_id        = trial_id,
                quat_wxyz       = quat_perturbed_wxyz,
                euler_noise_deg = euler_noise_deg,
            )
        )

    return results
