from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

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
    compute_box_composite5_normalized_moment_arm,
    compute_normalized_com_shift,
)

from mjcf_factory.box_composite5_mass_pattern_utiliy import generate_box_composite5_signed_mass_patterns_exponential


from box_composite5.spec import BoxCompositeMassSweepSpec


SLICE_RGBA_LIST = (
    (1.0, 0.0, 0.0, 1.0),   # red
    (1.0, 0.5, 0.0, 1.0),   # orange
    (1.0, 1.0, 0.0, 1.0),   # yellow
    (0.0, 0.8, 0.0, 1.0),   # green
    (0.0, 0.2, 1.0, 1.0),   # blue
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


@dataclass(frozen=True)
class BoxCompositeConditionAsset:
    object_name                     : str
    condition                       : str
    object_dir                      : Path

    stl_path                        : Path
    obj_path                        : Path
    xml_path                        : Path

    slice_masses                    : tuple[float, float, float, float, float]
    true_com_x                      : float
    com_shift_x                     : float
    normalized_shift_x              : float
    grasp_x                         : float
    moment_arm_x                    : float
    normalized_moment_arm_x         : float
    signed_normalized_moment_arm_x  : float


@dataclass(frozen=True)
class BoxCompositeObjectSet:
    spec                : BoxCompositeMassSweepSpec
    condition_assets    : dict[str, BoxCompositeConditionAsset]

    @property
    def object_name_list(self) -> list[str]:
        return [asset.object_name for asset in self.condition_assets.values()]


class BoxCompositeObjectSetGenerator:
    def __init__(self, spec: BoxCompositeMassSweepSpec):
        self.spec = spec

    def _build_object_name(
        self,
        condition: str,
    ) -> str:
        return f"box_composite5_{condition}"

    def _export_geometry_assets(
        self,
        object_dir: Path,
    ) -> dict[str, Path]:
        mesh_spec = BoxMeshSpec(
            model_name     = "textured",
            save_dir       = object_dir,
            size_xyz       = self.spec.full_size_xyz,
            export_formats = self.spec.export_formats,
        )
        exporter = BoxMeshExporter(mesh_spec)
        return exporter.export()

    def _export_mjcf_asset(
        self,
        object_dir   : Path,
        slice_masses : tuple[float, float, float, float, float],
    ) -> Path:
        mjcf_spec = BoxCompositeMJCFSpec(
            model_name      = "textured",
            save_dir        = object_dir,
            full_size_xyz   = self.spec.full_size_xyz,
            slice_masses    = slice_masses,
            slice_rgba_list = SLICE_RGBA_LIST,
        )
        exporter = BoxCompositeMJCFExporter(mjcf_spec)
        return exporter.export()

    def generate(self) -> BoxCompositeObjectSet:
        mass_patterns = generate_box_composite5_signed_mass_patterns_exponential(
            total_mass  = self.spec.total_mass,
            bias_levels = self.spec.bias_levels,
        )

        condition_assets = {}

        for condition in CONDITIONS:
            object_name = self._build_object_name(condition)
            object_dir  = self.spec.object_root_dir / object_name

            slice_masses   = mass_patterns[condition]
            geometry_paths = self._export_geometry_assets(
                object_dir = object_dir,
            )
            xml_path = self._export_mjcf_asset(
                object_dir   = object_dir,
                slice_masses = slice_masses,
            )

            com_result = compute_box_composite5_true_com(
                full_size_xyz = self.spec.full_size_xyz,
                slice_masses  = slice_masses,
            )
            moment_arm_result = compute_box_composite5_normalized_moment_arm(
                full_size_xyz = self.spec.full_size_xyz,
                true_com_x    = com_result.true_com_x,
                grasp_x       = self.spec.grasp_x,
            )
            normalized_shift_x = compute_normalized_com_shift(
                full_size_xyz = self.spec.full_size_xyz,
                com_shift_x   = com_result.com_shift_x,
                normalization = "length_x",
            )
            signed_normalized_moment_arm_x = (
                np.sign(moment_arm_result.moment_arm_x)
                * moment_arm_result.normalized_moment_arm_x
            )

            asset = BoxCompositeConditionAsset(
                object_name                    = object_name,
                condition                      = condition,
                object_dir                     = object_dir,
                stl_path                       = geometry_paths["stl"],
                obj_path                       = geometry_paths["obj"],
                xml_path                       = xml_path,
                slice_masses                   = slice_masses,
                true_com_x                     = com_result.true_com_x,
                com_shift_x                    = com_result.com_shift_x,
                normalized_shift_x             = normalized_shift_x,
                grasp_x                        = moment_arm_result.grasp_x,
                moment_arm_x                   = moment_arm_result.moment_arm_x,
                normalized_moment_arm_x        = moment_arm_result.normalized_moment_arm_x,
                signed_normalized_moment_arm_x = signed_normalized_moment_arm_x,
            )
            condition_assets[object_name] = asset

        return BoxCompositeObjectSet(
            spec             = self.spec,
            condition_assets = condition_assets,
        )
