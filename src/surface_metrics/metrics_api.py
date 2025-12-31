from __future__ import annotations
from typing import Dict, Optional
import numpy as np

from .transforms import transform_points_W_to_G
from .split_lr import split_left_right, SideSplit
from .local_geometry import compute_surface_metrics, fit_plane_pca, SurfaceMetrics

Array = np.ndarray


def compute_metrics_left_right(
    points_W: Array,
    normals_W: Optional[Array],
    patch_indices: Array,
    *,
    R_WG: Optional[Array] = None,
    t_WG: Optional[Array] = None,
    patch_points_G: Optional[Array] = None,
    lr_axis: int = 0,
    split_method: str = "kmeans",
    min_points_per_side: int = 30,
    seed: int = 0,
    # local neighborhood
    radius: Optional[float] = 0.005,
    k: Optional[int] = None,
    min_neighbors: int = 8,
    normal_var_use_abs_dot: bool = True,
    # whether to fit planes per side (debug/analysis)
    fit_side_planes: bool = True,
) -> Dict[str, object]:
    """
    High-level API:
      1) patch extraction (global indices)
      2) patch -> gripper frame (optional)
      3) split into L/R
      4) compute three metrics for each side

    Returns dict:
      - split: SideSplit
      - left/right: SurfaceMetrics
      - left_patch_indices_world/right_patch_indices_world
      - (optional) left_plane/right_plane in gripper frame
    """
    patch_indices = np.asarray(patch_indices, dtype=int).ravel()
    P_patch_W = points_W[patch_indices]

    if patch_points_G is None:
        if R_WG is None or t_WG is None:
            raise ValueError("Provide either (R_WG, t_WG) or patch_points_G.")
        P_patch_G = transform_points_W_to_G(P_patch_W, R_WG=R_WG, t_WG=t_WG)
    else:
        P_patch_G = np.asarray(patch_points_G, dtype=np.float64)
        if P_patch_G.shape != P_patch_W.shape:
            raise ValueError("patch_points_G must match patch_indices length and be (M,3).")

    split: SideSplit = split_left_right(
        P_patch_G,
        lr_axis=lr_axis,
        method=split_method,
        min_points_per_side=min_points_per_side,
        seed=seed,
    )

    left_world_idx = patch_indices[split.left_local_idx]
    right_world_idx = patch_indices[split.right_local_idx]

    left_metrics: SurfaceMetrics = compute_surface_metrics(
        points_W, normals_W,
        indices=left_world_idx,
        k=k, radius=radius,
        min_neighbors=min_neighbors,
        normal_var_use_abs_dot=normal_var_use_abs_dot,
    )
    right_metrics: SurfaceMetrics = compute_surface_metrics(
        points_W, normals_W,
        indices=right_world_idx,
        k=k, radius=radius,
        min_neighbors=min_neighbors,
        normal_var_use_abs_dot=normal_var_use_abs_dot,
    )

    out: Dict[str, object] = {
        "split": split,
        "left": left_metrics,
        "right": right_metrics,
        "left_patch_indices_world": left_world_idx,
        "right_patch_indices_world": right_world_idx,
    }

    if fit_side_planes:
        out["left_plane_G"] = fit_plane_pca(P_patch_G[split.left_local_idx]) if split.left_local_idx.size >= 3 else None
        out["right_plane_G"] = fit_plane_pca(P_patch_G[split.right_local_idx]) if split.right_local_idx.size >= 3 else None

    return out
