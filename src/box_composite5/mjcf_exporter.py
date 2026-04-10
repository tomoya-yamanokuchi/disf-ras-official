from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen = True)
class BoxCompositeMJCFSpec:
    model_name      : str
    save_dir        : Path
    full_size_xyz   : tuple[float, float, float]
    slice_masses    : tuple[float, float, float, float, float]
    slice_rgba_list : tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ] = (
        (1.0, 0.0, 0.0, 1.0),
        (1.0, 0.5, 0.0, 1.0),
        (1.0, 1.0, 0.0, 1.0),
        (0.0, 0.8, 0.0, 1.0),
        (0.0, 0.2, 1.0, 1.0),
    )
    geom_class : str = "object_collision"

    def __post_init__(self) -> None:
        if len(self.full_size_xyz) != 3:
            raise ValueError("full_size_xyz must have 3 elements.")
        if len(self.slice_masses) != 5:
            raise ValueError("slice_masses must have 5 elements.")
        if len(self.slice_rgba_list) != 5:
            raise ValueError("slice_rgba_list must have 5 elements.")
        if any(v <= 0 for v in self.full_size_xyz):
            raise ValueError("All full_size_xyz values must be positive.")
        if any(m <= 0 for m in self.slice_masses):
            raise ValueError("All slice masses must be positive.")


class BoxCompositeMJCFExporter:
    def __init__(
        self,
        spec : BoxCompositeMJCFSpec,
    ) -> None:
        self.spec = spec

    def _build_xml(self) -> str:
        lx, ly, lz = self.spec.full_size_xyz

        hx = lx / 10.0
        hy = ly / 2.0
        hz = lz / 2.0

        slice_width = lx / 5.0
        x_centers   = tuple(
            -lx / 2.0 + slice_width / 2.0 + i * slice_width
            for i in range(5)
        )

        geom_lines = []
        for i, (x, mass, rgba) in enumerate(
            zip(x_centers, self.spec.slice_masses, self.spec.slice_rgba_list)
        ):
            rgba_str = " ".join(f"{v:.6f}" for v in rgba)
            geom_lines.append(
                f'      <geom name="slice_{i}" type="box" '
                f'pos="{x:.6f} 0 0" '
                f'size="{hx:.6f} {hy:.6f} {hz:.6f}" '
                f'mass="{mass:.6f}" '
                f'rgba="{rgba_str}" '
                f'class="{self.spec.geom_class}"/>'
            )

        geom_block = "\n".join(geom_lines)

        return f"""<mujoco model="{self.spec.model_name}">
  <worldbody>
    <body name="{self.spec.model_name}">
      <freejoint/>
{geom_block}
    </body>
  </worldbody>
</mujoco>
"""

    def export(self) -> Path:
        xml_dir = self.spec.save_dir / "textured"
        xml_dir.mkdir(parents = True, exist_ok = True)

        xml_path = xml_dir / "textured.xml"
        xml_path.write_text(self._build_xml(), encoding = "utf-8")
        return xml_path
