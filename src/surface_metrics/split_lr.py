from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

Array = np.ndarray


def kmeans_1d_two_clusters(x: Array, iters: int = 30, seed: int = 0) -> Tuple[Array, Tuple[float, float]]:
    """Tiny 1D k-means (k=2). Returns labels (0/1) and centers."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size < 2:
        return np.zeros_like(x, dtype=int), (float(x.mean()) if x.size else 0.0, 0.0)

    c0, c1 = np.percentile(x, [25, 75]).astype(float)
    if np.isclose(c0, c1):
        c0, c1 = float(x.min()), float(x.max())

    for _ in range(iters):
        d0 = np.abs(x - c0)
        d1 = np.abs(x - c1)
        labels = (d1 < d0).astype(int)

        if np.all(labels == 0) or np.all(labels == 1):
            med = float(np.median(x))
            c0, c1 = med - 1e-6, med + 1e-6
            continue

        new_c0 = float(x[labels == 0].mean())
        new_c1 = float(x[labels == 1].mean())
        if np.isclose(new_c0, c0) and np.isclose(new_c1, c1):
            break
        c0, c1 = new_c0, new_c1

    return labels, (c0, c1)


@dataclass
class SideSplit:
    left_local_idx: Array   # indices into patch array (0..M-1)
    right_local_idx: Array
    # optional: centers for debugging
    centers: Optional[Tuple[float, float]] = None


def split_left_right(
    patch_points_G: Array,
    *,
    lr_axis: int = 0,
    method: str = "kmeans",        # "kmeans" or "sign"
    sign_threshold: float = 0.0,
    min_points_per_side: int = 30,
    seed: int = 0,
) -> SideSplit:
    """
    Split patch points (in gripper frame) into left/right.

    method:
      - "sign": coord < threshold => left
      - "kmeans": 1D kmeans on coord; smaller center => left
    """
    P = np.asarray(patch_points_G, dtype=np.float64)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError(f"patch_points_G must be (M,3), got {P.shape}")

    coord = P[:, lr_axis]

    centers = None
    if method == "sign":
        left_mask = coord < sign_threshold
        right_mask = ~left_mask
    elif method == "kmeans":
        labels, centers = kmeans_1d_two_clusters(coord, seed=seed)
        c0, c1 = centers
        if c0 <= c1:
            left_mask, right_mask = (labels == 0), (labels == 1)
        else:
            left_mask, right_mask = (labels == 1), (labels == 0)
    else:
        raise ValueError("method must be 'kmeans' or 'sign'")

    left_idx = np.nonzero(left_mask)[0]
    right_idx = np.nonzero(right_mask)[0]

    # fallback if split is too imbalanced
    if left_idx.size < min_points_per_side or right_idx.size < min_points_per_side:
        thr = float(np.median(coord))
        left_idx = np.nonzero(coord < thr)[0]
        right_idx = np.nonzero(coord >= thr)[0]

    return SideSplit(left_local_idx=left_idx, right_local_idx=right_idx, centers=centers)
