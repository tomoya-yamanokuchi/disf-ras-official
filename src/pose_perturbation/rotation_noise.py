from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from service import ExtendedRotation


@dataclass(frozen=True)
class RotationPerturbationConfig:
    max_angle_deg: float = 3.0
    euler_order  : str   = "xyz"


def apply_rotation_noise_to_quaternion_wxyz(
    base_quat_wxyz: np.ndarray,
    config        : RotationPerturbationConfig,
    rng           : np.random.Generator,
    left_multiply : bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    if base_quat_wxyz.shape != (4,):
        raise ValueError(
            f"base_quat_wxyz must have shape (4,), got {base_quat_wxyz.shape}"
        )

    base_rot = ExtendedRotation.from_quat(
        base_quat_wxyz,
    )

    euler_noise_deg = rng.uniform(
        low  = -config.max_angle_deg,
        high =  config.max_angle_deg,
        size = 3,
    )
    noise_rot = ExtendedRotation.from_euler(
        euler_noise_deg,
        degrees = True,
    )

    perturbed_rot : ExtendedRotation = noise_rot * base_rot if left_multiply else base_rot * noise_rot
    perturbed_quat_wxyz = perturbed_rot.as_quat_scalar_first()

    return perturbed_quat_wxyz, euler_noise_deg
