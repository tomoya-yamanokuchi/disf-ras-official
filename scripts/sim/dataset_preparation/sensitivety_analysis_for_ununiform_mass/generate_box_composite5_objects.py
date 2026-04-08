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
)


# ------------------------------------------------------------
# Experiment settings
# ------------------------------------------------------------
ROOT_DIR       = Path("./models/box_mesh")
FULL_SIZE_XYZ  = (0.10, 0.04, 0.04)
EXPORT_FORMATS = ("stl", "obj")

MASS_PATTERNS = {
    "uniform": (0.20, 0.20, 0.20, 0.20, 0.20),
    "mild"   : (0.16, 0.18, 0.20, 0.22, 0.24),
    "medium" : (0.12, 0.16, 0.20, 0.24, 0.28),
    "large"  : (0.08, 0.14, 0.20, 0.26, 0.32),
}

SLICE_RGBA_LIST = (
    (1.0, 0.0, 0.0, 0.5),   # red
    (1.0, 0.5, 0.0, 0.5),   # orange
    (1.0, 1.0, 0.0, 0.5),   # yellow
    (0.0, 0.8, 0.0, 0.5),   # green
    (0.0, 0.2, 1.0, 0.5),   # blue
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
) -> dict[str, object]:
    com_result = compute_box_composite5_true_com(
        full_size_xyz=full_size_xyz,
        slice_masses=slice_masses,
    )

    normalized_shift_x = compute_normalized_com_shift(
        full_size_xyz = full_size_xyz,
        com_shift_x   = com_result.com_shift_x,
        normalization = "length_x",
    )

    return {
        "condition": condition,
        "slice_masses": slice_masses,
        "slice_centers_x": com_result.slice_centers_x,
        "total_mass": com_result.total_mass,
        "true_com_x": com_result.true_com_x,
        "com_shift_x": com_result.com_shift_x,
        "normalized_shift_x": normalized_shift_x,
    }


def generate_single_condition(
    condition    : str,
    root_dir     : Path,
    full_size_xyz: tuple[float, float, float],
    export_formats: tuple[str, ...],
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
        for key, path in paths.items():
            print(f"  {key}: {path}")

        print("  --- CoM metadata ---")
        print(f"  condition          : {metadata['condition']}")
        print(f"  slice_masses       : {metadata['slice_masses']}")
        print(f"  slice_centers_x    : {metadata['slice_centers_x']}")
        print(f"  total_mass         : {metadata['total_mass']:.3f}")
        print(f"  true_com_x         : {metadata['true_com_x']:.3f}")
        print(f"  com_shift_x        : {metadata['com_shift_x']:.3f}")
        print(f"  normalized_shift_x : {metadata['normalized_shift_x']:.3f}")
        print()
