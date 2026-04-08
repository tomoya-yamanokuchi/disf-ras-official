from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import trimesh


@dataclass(frozen=True)
class BoxMeshSpec:
    """Specification for a box mesh to be exported."""
    model_name: str
    save_dir: Path
    size_xyz: tuple[float, float, float]
    export_formats: tuple[str, ...] = ("stl", "obj")
    centered_at_origin: bool = True

    def __post_init__(self) -> None:
        if len(self.size_xyz) != 3:
            raise ValueError("size_xyz must have exactly three elements: (x, y, z).")
        if any(s <= 0 for s in self.size_xyz):
            raise ValueError(f"All box sizes must be positive, got {self.size_xyz}.")
        if not self.model_name:
            raise ValueError("model_name must be a non-empty string.")
        if not self.export_formats:
            raise ValueError("export_formats must contain at least one format.")
        invalid = [fmt for fmt in self.export_formats if fmt not in {"stl", "obj"}]
        if invalid:
            raise ValueError(f"Unsupported export format(s): {invalid}")


@dataclass
class BoxMeshExporter:
    """Create and export a box mesh from a structured specification."""
    spec: BoxMeshSpec
    _mesh: trimesh.Trimesh | None = field(default=None, init=False, repr=False)

    def build_mesh(self) -> trimesh.Trimesh:
        """Build a trimesh box mesh from the spec."""
        mesh = trimesh.creation.box(extents=self.spec.size_xyz)

        if not self.spec.centered_at_origin:
            # By default trimesh box is centered at the origin.
            # This branch is kept for future extension.
            raise NotImplementedError(
                "Non-centered box placement is not implemented yet."
            )

        self._mesh = mesh
        return mesh

    @property
    def mesh(self) -> trimesh.Trimesh:
        """Return the current mesh, building it if necessary."""
        if self._mesh is None:
            return self.build_mesh()
        return self._mesh

    def export(self) -> dict[str, Path]:
        """Export the mesh in all requested formats."""
        self.spec.save_dir.mkdir(parents=True, exist_ok=True)

        exported_paths: dict[str, Path] = {}
        for fmt in self.spec.export_formats:
            # import ipdb; ipdb.set_trace()
            output_path = self.spec.save_dir / f"textured.{fmt}"
            self.mesh.export(output_path)
            exported_paths[fmt] = output_path

        return exported_paths
