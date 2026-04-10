from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen = True)
class SignedBiasLevels:
    mild   : float
    medium : float
    large  : float


def generate_box_composite5_mass_pattern_exponential(
    total_mass           : float,
    signed_bias_strength : float,
) -> tuple[float, float, float, float, float]:
    if total_mass <= 0:
        raise ValueError(f"total_mass must be positive, got {total_mass}.")

    positions  = (-2.0, -1.0, 0.0, 1.0, 2.0)
    weights    = [math.exp(signed_bias_strength * p) for p in positions]
    weight_sum = sum(weights)

    masses = tuple(total_mass * w / weight_sum for w in weights)
    return masses


def generate_box_composite5_signed_mass_patterns_exponential(
    total_mass  : float,
    bias_levels : SignedBiasLevels,
) -> dict[str, tuple[float, float, float, float, float]]:
    return {
        "large_left"   : generate_box_composite5_mass_pattern_exponential(total_mass, -bias_levels.large),
        "medium_left"  : generate_box_composite5_mass_pattern_exponential(total_mass, -bias_levels.medium),
        "mild_left"    : generate_box_composite5_mass_pattern_exponential(total_mass, -bias_levels.mild),
        "uniform"      : generate_box_composite5_mass_pattern_exponential(total_mass,  0.0),
        "mild_right"   : generate_box_composite5_mass_pattern_exponential(total_mass,  bias_levels.mild),
        "medium_right" : generate_box_composite5_mass_pattern_exponential(total_mass,  bias_levels.medium),
        "large_right"  : generate_box_composite5_mass_pattern_exponential(total_mass,  bias_levels.large),
    }
