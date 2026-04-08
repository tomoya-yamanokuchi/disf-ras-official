from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BoxCompositeMJCFSpec:
    model_name   : str
    save_dir     : Path
    full_size_xyz: tuple[float, float, float]
    slice_masses : tuple[float, float, float, float, float]
    rgba         : tuple[float, float, float, float] = (0.7, 0.7, 0.7, 1.0)
    geom_class   : str = "object"

    def __post_init__(self) -> None:
        if len(self.full_size_xyz) != 3:
            raise ValueError("full_size_xyz must have 3 elements.")
        if len(self.slice_masses) != 5:
            raise ValueError("slice_masses must have 5 elements.")
        if any(v <= 0 for v in self.full_size_xyz):
            raise ValueError("All full_size_xyz values must be positive.")
        if any(m <= 0 for m in self.slice_masses):
            raise ValueError("All slice masses must be positive.")


class BoxCompositeMJCFExporter:
    def __init__(self, spec: BoxCompositeMJCFSpec) -> None:
        self.spec = spec

    def _build_xml(self) -> str:
        lx, ly, lz = self.spec.full_size_xyz
        hx = lx / 10.0
        hy = ly / 2.0
        hz = lz / 2.0

        x_centers = (
            -2 * lx / 5.0,
            -1 * lx / 5.0,
             0.0,
             1 * lx / 5.0,
             2 * lx / 5.0,
        )
        # For lx=0.10, this becomes (-0.04, -0.02, 0.0, 0.02, 0.04)

        rgba = " ".join(str(v) for v in self.spec.rgba)

        geom_lines = []
        for i, (x, mass) in enumerate(zip(x_centers, self.spec.slice_masses)):
            geom_lines.append(
                f'      <geom name="slice_{i}" type="box" '
                f'pos="{x:.6f} 0 0" size="{hx:.6f} {hy:.6f} {hz:.6f}" '
                f'mass="{mass:.6f}" rgba="{rgba}" class="{self.spec.geom_class}"/>'
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
        xml_dir.mkdir(parents=True, exist_ok=True)

        xml_path = xml_dir / "textured.xml"
        xml_path.write_text(self._build_xml(), encoding="utf-8")
        return xml_path
