from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from mesh_factory.box_mesh_exporter import (
    BoxMeshExporter,
    BoxMeshSpec,
)

from .mass_pattern import (
    SignedBiasLevels,
    generate_box_composite5_signed_mass_patterns_exponential,
)
from .metrics import (
    compute_box_composite5_true_com,
    compute_box_composite5_normalized_moment_arm,
)
from .mjcf_exporter import (
    BoxCompositeMJCFExporter,
    BoxCompositeMJCFSpec,
)
from .spec import BoxComposite5PatternSpec


SLICE_RGBA_LIST = (
    (1.0, 0.0, 0.0, 1.0),
    (1.0, 0.5, 0.0, 1.0),
    (1.0, 1.0, 0.0, 1.0),
    (0.0, 0.8, 0.0, 1.0),
    (0.0, 0.2, 1.0, 1.0),
)


@dataclass(frozen = True)
class BoxComposite5PatternAsset:
    object_name                     : str
    object_dir                      : Path
    slice_masses                    : tuple[float, float, float, float, float]
    signed_normalized_moment_arm_x  : float


class BoxComposite5PatternAssetBuilder:
    def __init__(
        self,
        spec : BoxComposite5PatternSpec,
    ) -> None:
        self.spec = spec

    def _build_object_name(self) -> str:
        return f"box_composite5_{self.spec.condition}"

    def _build_slice_masses(self) -> tuple[float, float, float, float, float]:
        bias_levels = SignedBiasLevels(
            mild   = self.spec.bias_mild,
            medium = self.spec.bias_medium,
            large  = self.spec.bias_large,
        )
        mass_patterns = generate_box_composite5_signed_mass_patterns_exponential(
            total_mass  = self.spec.total_mass,
            bias_levels = bias_levels,
        )
        return mass_patterns[self.spec.condition]

    def _export_geometry_assets(
        self,
        object_dir    : Path,
    ) -> None:
        mesh_spec = BoxMeshSpec(
            model_name     = "textured",
            save_dir       = object_dir,
            size_xyz       = self.spec.full_size_xyz,
            export_formats = ("stl", "obj"),
        )
        exporter = BoxMeshExporter(mesh_spec)
        exporter.export()

    def _export_mjcf_asset(
        self,
        object_dir    : Path,
        slice_masses  : tuple[float, float, float, float, float],
    ) -> None:
        mjcf_spec = BoxCompositeMJCFSpec(
            model_name      = "textured",
            save_dir        = object_dir,
            full_size_xyz   = self.spec.full_size_xyz,
            slice_masses    = slice_masses,
            slice_rgba_list = SLICE_RGBA_LIST,
        )
        exporter = BoxCompositeMJCFExporter(mjcf_spec)
        exporter.export()

    def _compute_signed_normalized_moment_arm_x(
        self,
        slice_masses : tuple[float, float, float, float, float],
    ) -> float:
        com_result = compute_box_composite5_true_com(
            full_size_xyz = self.spec.full_size_xyz,
            slice_masses  = slice_masses,
        )
        moment_arm_result = compute_box_composite5_normalized_moment_arm(
            full_size_xyz = self.spec.full_size_xyz,
            true_com_x    = com_result.true_com_x,
            grasp_x       = self.spec.grasp_x,
        )
        return float(
            np.sign(moment_arm_result.moment_arm_x)
            * moment_arm_result.normalized_moment_arm_x
        )

    def build(self) -> BoxComposite5PatternAsset:
        object_name  = self._build_object_name()
        object_dir   = self.spec.object_root_dir / object_name
        slice_masses = self._build_slice_masses()

        self._export_geometry_assets(
            object_dir = object_dir,
        )
        self._export_mjcf_asset(
            object_dir   = object_dir,
            slice_masses = slice_masses,
        )

        signed_normalized_moment_arm_x = self._compute_signed_normalized_moment_arm_x(
            slice_masses = slice_masses,
        )

        return BoxComposite5PatternAsset(
            object_name                    = object_name,
            object_dir                     = object_dir,
            slice_masses                   = slice_masses,
            signed_normalized_moment_arm_x = signed_normalized_moment_arm_x,
        )
