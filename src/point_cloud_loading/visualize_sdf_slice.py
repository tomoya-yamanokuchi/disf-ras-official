import matplotlib.pyplot as plt
import numpy as np


def visualize_sdf_slice(self, axis: str = "z", index: int | None = None):
    """
    SDF の1スライスを matplotlib で表示。
    axis: "x" / "y" / "z" のどれか
    index: スライス番号（None のときは中央）
    """
    sdf = self.sdf
    Nx, Ny, Nz = sdf.shape

    if axis == "z":
        if index is None:
            index = Nz // 2
        slice_data = sdf[:, :, index]          # (Nx, Ny)
        xlabel, ylabel = "X", "Y"

    elif axis == "y":
        if index is None:
            index = Ny // 2
        slice_data = sdf[:, index, :]          # (Nx, Nz)
        xlabel, ylabel = "X", "Z"

    elif axis == "x":
        if index is None:
            index = Nx // 2
        slice_data = sdf[index, :, :]          # (Ny, Nz)
        xlabel, ylabel = "Y", "Z"

    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")

    plt.figure(figsize=(6, 5))
    im = plt.imshow(
        slice_data.T,           # 軸を視覚的に分かりやすくするため転置
        origin="lower",
        cmap="viridis"
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"SDF slice axis={axis}, index={index}")
    plt.colorbar(im, label="distance [m?]")
    plt.tight_layout()
    plt.show()
