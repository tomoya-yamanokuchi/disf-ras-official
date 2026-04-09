from __future__ import annotations


from mjcf_factory.box_composite5_mass_pattern_utiliy import (
    generate_box_composite5_signed_mass_patterns,
    SignedBiasLevels,
)


if __name__ == "__main__":
    patterns = generate_box_composite5_signed_mass_patterns(
        total_mass  = 0.5,
        bias_levels = SignedBiasLevels(
            mild   = 0.10,
            medium = 0.20,
            large  = 0.40,
        ),
    )

    for name, masses in patterns.items():
        print(name, ":", masses)
