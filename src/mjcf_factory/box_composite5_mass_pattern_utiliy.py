from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SignedBiasLevels:
    mild  : float
    medium: float
    large : float


def generate_box_composite5_mass_pattern(
    total_mass: float,
    signed_bias_strength: float,
) -> tuple[float, float, float, float, float]:
    """
    Generate a 5-slice mass pattern for a composite box.

    Parameters
    ----------
    total_mass:
        Total mass of the object.
    signed_bias_strength:
        Signed bias strength.
        - 0.0  -> uniform
        - >0.0 -> right-heavy
        - <0.0 -> left-heavy

    Returns
    -------
    tuple[float, float, float, float, float]
        Masses for 5 slices ordered from left to right.

    Notes
    -----
    The unnormalized weights are:
        [1 - 2s, 1 - s, 1, 1 + s, 1 + 2s]

    To ensure all slice masses remain positive, |s| must be < 0.5.
    """
    if total_mass <= 0:
        raise ValueError(f"total_mass must be positive, got {total_mass}.")
    if abs(signed_bias_strength) >= 0.5:
        raise ValueError(
            "signed_bias_strength must satisfy abs(signed_bias_strength) < 0.5 "
            "to keep all slice masses positive."
        )

    s = signed_bias_strength
    weights = [
        1.0 - 2.0 * s,
        1.0 - 1.0 * s,
        1.0,
        1.0 + 1.0 * s,
        1.0 + 2.0 * s,
    ]
    print(weights)

    weight_sum = sum(weights)
    masses = tuple(total_mass * w / weight_sum for w in weights)

    # import ipdb; ipdb.set_trace()
    return masses




def generate_box_composite5_signed_mass_patterns(
    total_mass: float,
    bias_levels: SignedBiasLevels,
) -> dict[str, tuple[float, float, float, float, float]]:
    """
    Generate 7 signed mass-pattern conditions:
        large_left, medium_left, mild_left, uniform,
        mild_right, medium_right, large_right
    """
    return {
        "large_left":   generate_box_composite5_mass_pattern(total_mass, -bias_levels.large),
        "medium_left":  generate_box_composite5_mass_pattern(total_mass, -bias_levels.medium),
        "mild_left":    generate_box_composite5_mass_pattern(total_mass, -bias_levels.mild),
        "uniform":      generate_box_composite5_mass_pattern(total_mass, 0.0),
        "mild_right":   generate_box_composite5_mass_pattern(total_mass,  bias_levels.mild),
        "medium_right": generate_box_composite5_mass_pattern(total_mass,  bias_levels.medium),
        "large_right":  generate_box_composite5_mass_pattern(total_mass,  bias_levels.large),
    }


import math
def generate_box_composite5_mass_pattern_exponential(
    total_mass          : float,
    signed_bias_strength: float,
) -> tuple[float, float, float, float, float]:
    """
    Generate a 5-slice mass pattern using exponential weighting.

    Parameters
    ----------
    total_mass:
        Total mass of the object.
    signed_bias_strength:
        Signed exponential bias strength.
        - 0.0  -> uniform
        - >0.0 -> right-heavy
        - <0.0 -> left-heavy

    Returns
    -------
    tuple[float, float, float, float, float]
        Masses ordered from left to right.
    """
    if total_mass <= 0:
        raise ValueError(f"total_mass must be positive, got {total_mass}.")

    positions = (-2.0, -1.0, 0.0, 1.0, 2.0)
    weights = [math.exp(signed_bias_strength * p) for p in positions]
    weight_sum = sum(weights)

    # print("weights = ", weights)

    masses = tuple(total_mass * w / weight_sum for w in weights)

    print("masses = ", masses)

    return masses


def generate_box_composite5_signed_mass_patterns_exponential(
    total_mass: float,
    bias_levels: SignedBiasLevels,
) -> dict[str, tuple[float, float, float, float, float]]:
    return {
        "large_left":   generate_box_composite5_mass_pattern_exponential(total_mass, -bias_levels.large),
        "medium_left":  generate_box_composite5_mass_pattern_exponential(total_mass, -bias_levels.medium),
        "mild_left":    generate_box_composite5_mass_pattern_exponential(total_mass, -bias_levels.mild),
        "uniform":      generate_box_composite5_mass_pattern_exponential(total_mass, 0.0),
        "mild_right":   generate_box_composite5_mass_pattern_exponential(total_mass,  bias_levels.mild),
        "medium_right": generate_box_composite5_mass_pattern_exponential(total_mass,  bias_levels.medium),
        "large_right":  generate_box_composite5_mass_pattern_exponential(total_mass,  bias_levels.large),
    }
