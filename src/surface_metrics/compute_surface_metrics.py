from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np


Array = np.ndarray


def transform_points_to_frame(points_W: Array, R_WG: Array, t_WG: Array) -> Array:
    """
    World -> Gripper frame (G) transform.
    R_WG: (3,3) rotation from G to W
    t_WG: (3,)  translation of G origin in W
    returns points_G = R_GW (points_W - t_WG)
    """
    points_W = np.asarray(points_W, dtype=np.float64)
    R_WG = np.asarray(R_WG, dtype=np.float64)
    t_WG = np.asarray(t_WG, dtype=np.float64).reshape(3)
    R_GW = R_WG.T
    return (points_W - t_WG[None, :]) @ R_GW.T


def _kmeans_1d_two_clusters(x: Array, iters: int = 30, seed: int = 0) -> Tuple[Array, Tuple[float, float]]:
    """
    Very small 1D k-means (k=2). Returns labels (0/1) and centers.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size < 2:
        return np.zeros_like(x, dtype=int), (float(x.mean()) if x.size else 0.0, 0.0)

    # init: pick two quantiles as centers
    c0, c1 = np.percentile(x, [25, 75]).astype(float)
    if np.isclose(c0, c1):
        # fallback: random split
        c0 = float(x.min())
        c1 = float(x.max())

    for _ in range(iters):
        d0 = np.abs(x - c0)
        d1 = np.abs(x - c1)
        labels = (d1 < d0).astype(int)

        if np.all(labels == 0) or np.all(labels == 1):
            # degenerate; jitter centers
            c0 = float(np.median(x) - 1e-6)
            c1 = float(np.median(x) + 1e-6)
            continue

        new_c0 = float(x[labels == 0].mean())
        new_c1 = float(x[labels == 1].mean())
        if np.isclose(new_c0, c0) and np.isclose(new_c1, c1):
            break
        c0, c1 = new_c0, new_c1

    return labels, (c0, c1)


def fit_plane_pca(points: Array, eps: float = 1e-12) -> Tuple[Array, float]:
    """
    Fit a plane via PCA: returns (n, d) such that n^T x + d = 0.
    n is unit normal.
    """
    P = np.asarray(points, dtype=np.float64)
    mu = P.mean(axis=0)
    X = P - mu[None, :]
    C = (X.T @ X) / max(P.shape[0], 1)
    w, V = np.linalg.eigh(C)  # ascending
    n = V[:, 0]
    n = n / max(np.linalg.norm(n), eps)
    d = -float(n @ mu)
    return n, d


@dataclass
class SideSplitResult:
    left_idx: Array          # indices into the *patch list* (0..M-1)
    right_idx: Array
    # optional fitted planes in gripper frame
    left_plane: Optional[Tuple[Array, float]]
    right_plane: Optional[Tuple[Array, float]]


def split_patch_left_right(
    patch_points_G: Array,
    *,
    lr_axis: int = 0,                 # 0:x, 1:y, 2:z in gripper frame
    method: str = "kmeans",           # "sign" or "kmeans"
    sign_threshold: float = 0.0,      # used when method="sign"
    min_points_per_side: int = 30,
    fit_planes: bool = True,
    seed: int = 0,
) -> SideSplitResult:
    """
    Split patch points into left/right sets in gripper frame.

    patch_points_G: (M,3) points already in gripper frame.
    lr_axis: axis that separates left/right (typically x).
    method:
      - "sign": left if coord < threshold, right otherwise
      - "kmeans": 1D kmeans on coord to get two clusters
    """
    P = np.asarray(patch_points_G, dtype=np.float64)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError(f"patch_points_G must be (M,3), got {P.shape}")

    coord = P[:, lr_axis]

    if method == "sign":
        left_mask = coord < sign_threshold
        right_mask = ~left_mask
    elif method == "kmeans":
        labels, (c0, c1) = _kmeans_1d_two_clusters(coord, seed=seed)
        # smaller center => left, larger center => right
        if c0 <= c1:
            left_mask = labels == 0
            right_mask = labels == 1
        else:
            left_mask = labels == 1
            right_mask = labels == 0
    else:
        raise ValueError("method must be 'sign' or 'kmeans'")

    left_idx = np.nonzero(left_mask)[0]
    right_idx = np.nonzero(right_mask)[0]

    # sanity: ensure both sides have enough points
    if left_idx.size < min_points_per_side or right_idx.size < min_points_per_side:
        # fallback: median sign split (often fixes weird init / origin shifts)
        thr = float(np.median(coord))
        left_idx = np.nonzero(coord < thr)[0]
        right_idx = np.nonzero(coord >= thr)[0]

    left_plane = right_plane = None
    if fit_planes:
        if left_idx.size >= 3:
            left_plane = fit_plane_pca(P[left_idx])
        if right_idx.size >= 3:
            right_plane = fit_plane_pca(P[right_idx])

    return SideSplitResult(
        left_idx=left_idx,
        right_idx=right_idx,
        left_plane=left_plane,
        right_plane=right_plane,
    )


# ---- main wrapper: compute metrics per side ----
def compute_surface_metrics_per_side(
    points_W: Array,
    normals_W: Optional[Array],
    patch_indices: Array,
    *,
    # Provide either:
    # 1) already-in-gripper patch points by giving R_WG,t_WG (world->gripper),
    # or 2) precomputed patch_points_G directly (pass None for R_WG/t_WG and set patch_points_G)
    R_WG: Optional[Array] = None,
    t_WG: Optional[Array] = None,
    patch_points_G: Optional[Array] = None,
    lr_axis: int = 0,
    split_method: str = "kmeans",
    radius: float = 0.005,     # 5mm neighborhood (example)
    k: Optional[int] = None,   # if you prefer kNN, set k and set radius=None in compute_surface_metrics
    min_neighbors: int = 8,
    min_points_per_side: int = 30,
    seed: int = 0,
):
    """
    Returns:
      dict with:
        - "split": SideSplitResult
        - "left":  SurfaceMetrics
        - "right": SurfaceMetrics
        - "left_patch_indices_world": indices into original points array
        - "right_patch_indices_world": indices into original points array
    Requires compute_surface_metrics() defined previously.
    """
    patch_indices = np.asarray(patch_indices, dtype=int).ravel()
    P_patch_W = points_W[patch_indices]

    if patch_points_G is None:
        if R_WG is None or t_WG is None:
            raise ValueError("Provide either (R_WG, t_WG) or patch_points_G.")
        P_patch_G = transform_points_to_frame(P_patch_W, R_WG=R_WG, t_WG=t_WG)
    else:
        P_patch_G = np.asarray(patch_points_G, dtype=np.float64)
        if P_patch_G.shape[0] != patch_indices.size:
            raise ValueError("patch_points_G must correspond 1-to-1 with patch_indices.")

    split = split_patch_left_right(
        P_patch_G,
        lr_axis=lr_axis,
        method=split_method,
        min_points_per_side=min_points_per_side,
        fit_planes=True,
        seed=seed,
    )

    left_patch_indices_world = patch_indices[split.left_idx]
    right_patch_indices_world = patch_indices[split.right_idx]

    # call your previous function
    left_metrics = compute_surface_metrics(
        points_W,
        normals_W,
        radius=radius,
        k=k,
        indices=left_patch_indices_world,
        min_neighbors=min_neighbors,
    )
    right_metrics = compute_surface_metrics(
        points_W,
        normals_W,
        radius=radius,
        k=k,
        indices=right_patch_indices_world,
        min_neighbors=min_neighbors,
    )

    return {
        "split": split,
        "left": left_metrics,
        "right": right_metrics,
        "left_patch_indices_world": left_patch_indices_world,
        "right_patch_indices_world": right_patch_indices_world,
    }
