from __future__ import annotations

from pathlib import Path

from mesh_factory.box_mesh_exporter import (
    BoxMeshExporter,
    BoxMeshSpec,
)
from mjcf_factory.box_composite_xml_exporter import (
    BoxCompositeMJCFExporter,
    BoxCompositeMJCFSpec,
)

from mjcf_factory.box_composite_mass_utils import (
    compute_box_composite5_true_com,
    compute_normalized_com_shift,
    compute_box_composite5_normalized_moment_arm,
)


from mjcf_factory.box_composite5_mass_pattern_utiliy import (
    generate_box_composite5_signed_mass_patterns,
    generate_box_composite5_signed_mass_patterns_exponential,
    SignedBiasLevels,
)

# ------------------------------------------------------------
# Experiment settings
# ------------------------------------------------------------
ROOT_DIR       = Path("./models/box_mesh")
EXPORT_FORMATS = ("stl", "obj")

FULL_SIZE_XYZ  = (0.20, 0.04, 0.04)
GRASP_X = -0.08

# --------------------------
# User-controlled parameters
# --------------------------
TOTAL_MASS = 0.5 # 0.75 # 1.0 # 0.50
# TOTAL_MASS = 0.7 # 0.75 # 1.0 # 0.50
# TOTAL_MASS = 1.0 # 0.75 # 1.0 # 0.50

BIAS_LEVELS = SignedBiasLevels(
    mild   = 0.5,
    medium = 1.0,
    large  = 1.5,
)


# Automatically generated 7 signed conditions
MASS_PATTERNS = generate_box_composite5_signed_mass_patterns_exponential(
    total_mass  = TOTAL_MASS,
    bias_levels = BIAS_LEVELS,
)

CONDITIONS = (
    "large_left",
    "medium_left",
    "mild_left",
    "uniform",
    "mild_right",
    "medium_right",
    "large_right",
)

SLICE_RGBA_LIST = (
    (1.0, 0.0, 0.0, 1.0),   # red
    (1.0, 0.5, 0.0, 1.0),   # orange
    (1.0, 1.0, 0.0, 1.0),   # yellow
    (0.0, 0.8, 0.0, 1.0),   # green
    (0.0, 0.2, 1.0, 1.0),   # blue
)


def build_object_name(condition: str) -> str:
    return f"box_composite5_{condition}"


def export_geometry_assets(
    object_dir    : Path,
    full_size_xyz : tuple[float, float, float],
    export_formats: tuple[str, ...],
) -> dict[str, Path]:
    """
    Export the external geometry used for grasp planning.
    The geometry is identical across all mass-shift conditions.
    """
    spec = BoxMeshSpec(
        model_name     = "textured",
        save_dir       = object_dir,
        size_xyz       = full_size_xyz,
        export_formats = export_formats,
    )
    exporter = BoxMeshExporter(spec)
    return exporter.export()


def export_mjcf_asset(
    object_dir   : Path,
    object_name  : str,
    full_size_xyz: tuple[float, float, float],
    slice_masses : tuple[float, float, float, float, float],
) -> Path:
    """
    Export the MuJoCo XML used for physics simulation.
    The mass pattern changes across conditions while the outer geometry remains fixed.
    """
    spec = BoxCompositeMJCFSpec(
        model_name      = "textured",
        save_dir        = object_dir,
        full_size_xyz   = full_size_xyz,
        slice_masses    = slice_masses,
        geom_class      = "object_collision",
        slice_rgba_list = SLICE_RGBA_LIST,
    )
    exporter = BoxCompositeMJCFExporter(spec)
    return exporter.export()


def compute_condition_metadata(
    condition    : str,
    full_size_xyz: tuple[float, float, float],
    slice_masses : tuple[float, float, float, float, float],
    grasp_x      : float,
) -> dict[str, object]:
    com_result = compute_box_composite5_true_com(
        full_size_xyz = full_size_xyz,
        slice_masses  = slice_masses,
    )

    normalized_shift_x = compute_normalized_com_shift(
        full_size_xyz = full_size_xyz,
        com_shift_x   = com_result.com_shift_x,
        normalization = "length_x",
    )

    moment_arm_result = compute_box_composite5_normalized_moment_arm(
        full_size_xyz = full_size_xyz,
        true_com_x    = com_result.true_com_x,
        grasp_x       = grasp_x,
    )

    return {
        "condition"              : condition,
        "slice_masses"           : slice_masses,
        "slice_centers_x"        : com_result.slice_centers_x,
        "total_mass"             : com_result.total_mass,
        "true_com_x"             : com_result.true_com_x,
        "com_shift_x"            : com_result.com_shift_x,
        "normalized_shift_x"     : normalized_shift_x,
        "grasp_x"                : moment_arm_result.grasp_x,
        "moment_arm_x"           : moment_arm_result.moment_arm_x,
        "abs_moment_arm_x"       : moment_arm_result.abs_moment_arm_x,
        "normalized_moment_arm_x": moment_arm_result.normalized_moment_arm_x,
    }


def generate_single_condition(
    condition     : str,
    root_dir      : Path,
    full_size_xyz : tuple[float, float, float],
    export_formats: tuple[str, ...],
    grasp_x       : float,
) -> dict[str, Path]:
    if condition not in MASS_PATTERNS:
        raise ValueError(f"Unknown condition: {condition}")

    object_name = build_object_name(condition)
    object_dir = root_dir / object_name

    geometry_paths = export_geometry_assets(
        object_dir     = object_dir,
        full_size_xyz  = full_size_xyz,
        export_formats = export_formats,
    )

    xml_path = export_mjcf_asset(
        object_dir    = object_dir,
        object_name   = object_name,
        full_size_xyz = full_size_xyz,
        slice_masses  = MASS_PATTERNS[condition],
    )

    metadata = compute_condition_metadata(
        condition     = condition,
        full_size_xyz = full_size_xyz,
        slice_masses  = MASS_PATTERNS[condition],
        grasp_x       = grasp_x,
    )

    return {
        "object_name": object_name,
        "paths"      : {
            **geometry_paths,
            "xml": xml_path,
        },
        "metadata"   : metadata,
    }


def generate_all_conditions() -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}

    for condition in MASS_PATTERNS:
        result = generate_single_condition(
            condition     = condition,
            root_dir      = ROOT_DIR,
            full_size_xyz = FULL_SIZE_XYZ,
            export_formats= EXPORT_FORMATS,
            grasp_x       = GRASP_X,
        )
        results[result["object_name"]] = result

    return results


if __name__ == "__main__":
    results = generate_all_conditions()

    print("=" * 100)
    print("Generated box composite objects and computed true CoM metadata")
    print("=" * 100)

    for object_name, result in results.items():
        paths = result["paths"]
        metadata = result["metadata"]

        print(f"[{object_name}]")
        # for key, path in paths.items():
        #     print(f"  {key}: {path}")

        print("  --- CoM metadata ---")
        print(f"  condition          : {metadata['condition']}")
        # print(f"  slice_masses       : {metadata['slice_masses']}")
        # print(f"  slice_centers_x    : {metadata['slice_centers_x']}")
        # print(f"  total_mass         : {metadata['total_mass']:.6f}")
        # print(f"  true_com_x         : {metadata['true_com_x']:.6f}")
        # print(f"  com_shift_x        : {metadata['com_shift_x']:.6f}")
        # print(f"  normalized_shift_x : {metadata['normalized_shift_x']:.6f}")
        # print("  --- grasp-relative metadata ---")
        # print(f"  grasp_x                   : {metadata['grasp_x']:.6f}")
        # print(f"  moment_arm_x              : {metadata['moment_arm_x']:.6f}")
        # print(f"  abs_moment_arm_x          : {metadata['abs_moment_arm_x']:.6f}")
        print(f"  normalized_moment_arm_x   : {metadata['normalized_moment_arm_x']:.6f}")
        print()
