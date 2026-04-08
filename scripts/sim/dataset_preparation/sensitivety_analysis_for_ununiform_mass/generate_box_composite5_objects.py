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


# ------------------------------------------------------------
# Experiment settings
# ------------------------------------------------------------
ROOT_DIR = Path("./models/box_mesh")
FULL_SIZE_XYZ = (0.10, 0.04, 0.04)
EXPORT_FORMATS = ("stl", "obj")

MASS_PATTERNS = {
    "uniform": (0.20, 0.20, 0.20, 0.20, 0.20),
    "mild":    (0.16, 0.18, 0.20, 0.22, 0.24),
    "medium":  (0.12, 0.16, 0.20, 0.24, 0.28),
    "large":   (0.08, 0.14, 0.20, 0.26, 0.32),
}


def build_object_name(condition: str) -> str:
    return f"box_composite5_{condition}"


def export_geometry_assets(
    object_dir: Path,
    full_size_xyz: tuple[float, float, float],
    export_formats: tuple[str, ...],
) -> dict[str, Path]:
    """
    Export the external geometry used for grasp planning.
    The geometry is identical across all mass-shift conditions.
    """
    spec = BoxMeshSpec(
        model_name="textured",
        save_dir=object_dir,
        size_xyz=full_size_xyz,
        export_formats=export_formats,
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
        model_name    = "textured",
        save_dir      = object_dir,
        full_size_xyz = full_size_xyz,
        slice_masses  = slice_masses,
        geom_class    = "object_collision"
    )
    exporter = BoxCompositeMJCFExporter(spec)
    return exporter.export()


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

    result = dict(geometry_paths)
    result["xml"] = xml_path
    return result


def generate_all_conditions() -> dict[str, dict[str, Path]]:
    results: dict[str, dict[str, Path]] = {}

    for condition in MASS_PATTERNS:
        object_name = build_object_name(condition)
        results[object_name] = generate_single_condition(
            condition      = condition,
            root_dir       = ROOT_DIR,
            full_size_xyz  = FULL_SIZE_XYZ,
            export_formats = EXPORT_FORMATS,
        )

    return results


if __name__ == "__main__":
    results = generate_all_conditions()

    print("=" * 80)
    print("Generated box composite objects:")
    print("=" * 80)

    for object_name, paths in results.items():
        print(f"[{object_name}]")
        for key, path in paths.items():
            print(f"  {key}: {path}")
        print()
