from pathlib import Path

from mesh_factory.box_mesh_exporter import BoxMeshExporter, BoxMeshSpec


CONDITIONS = (
    "uniform",
    "mild",
    "medium",
    "large",
)


def build_object_name(condition: str) -> str:
    return f"box_composite5_{condition}"


def export_box_meshes_for_all_conditions(
    root_dir      : Path,
    size_xyz      : tuple[float, float, float],
    export_formats: tuple[str, ...] = ("stl", "obj"),
) -> dict[str, dict[str, Path]]:
    results: dict[str, dict[str, Path]] = {}

    for condition in CONDITIONS:
        object_name = build_object_name(condition)

        spec = BoxMeshSpec(
            model_name     = "textured",
            save_dir       = root_dir / object_name,
            size_xyz       = size_xyz,
            export_formats = export_formats,
        )

        exporter = BoxMeshExporter(spec)
        results[object_name] = exporter.export()

    return results


if __name__ == "__main__":
    results = export_box_meshes_for_all_conditions(
        root_dir       = Path("./models/box_mesh"),
        size_xyz       = (0.10, 0.04, 0.04),
        export_formats = ("stl", "obj"),
    )

    for object_name, paths in results.items():
        print(f"[{object_name}]")
        for fmt, path in paths.items():
            print(f"  {fmt}: {path}")
