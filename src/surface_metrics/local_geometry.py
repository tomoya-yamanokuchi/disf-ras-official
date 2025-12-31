from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

try:
    from scipy.spatial import cKDTree
except ImportError as e:
    raise ImportError("local_geometry.py requires scipy (scipy.spatial.cKDTree).") from e

Array = np.ndarray


def summarize_1d(x: Array, name: str) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {f"{name}_count": 0.0}
    return {
        f"{name}_count": float(x.size),
        f"{name}_mean": float(np.mean(x)),
        f"{name}_median": float(np.median(x)),
        f"{name}_std": float(np.std(x)),
        f"{name}_p90": float(np.percentile(x, 90)),
        f"{name}_p95": float(np.percentile(x, 95)),
        f"{name}_p99": float(np.percentile(x, 99)),
        f"{name}_min": float(np.min(x)),
        f"{name}_max": float(np.max(x)),
    }


def fit_plane_pca(points: Array, eps: float = 1e-12):
    """
    PCA plane fit: returns (n, d) where n^T x + d = 0, n is unit normal.
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
class SurfaceMetrics:
    plane_rms: Array              # (N,) NaN outside indices
    curvature: Array              # (N,) NaN outside indices
    normal_var: Optional[Array]   # (N,) NaN outside indices / None if normals not provided
    n_fit: Array                  # (N,3) NaN outside indices
    neighbor_count: Array         # (N,)
    summary: Dict[str, float]     # aggregated stats


def compute_surface_metrics(
    points: Array,
    normals: Optional[Array] = None,
    *,
    indices: Optional[Array] = None,
    k: Optional[int] = 30,
    radius: Optional[float] = None,
    normal_var_use_abs_dot: bool = True,
    min_neighbors: int = 8,
    eps: float = 1e-12,
) -> SurfaceMetrics:
    """
    Metrics:
      - plane_rms: RMS distance to local PCA plane
      - curvature: lambda_min / trace
      - normal_var: mean(1 - dot) (or 1 - abs(dot)) between point normal and neighbors
    """
    P = np.asarray(points, dtype=np.float64)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError(f"points must be (N,3), got {P.shape}")
    N = P.shape[0]

    N_in = normals
    if N_in is not None:
        N_in = np.asarray(N_in, dtype=np.float64)
        if N_in.shape != (N, 3):
            raise ValueError(f"normals must be (N,3), got {N_in.shape}")
        n_norm = np.linalg.norm(N_in, axis=1, keepdims=True)
        N_in = N_in / np.maximum(n_norm, eps)

    if radius is None:
        if k is None or k < 2:
            raise ValueError("Use k>=2 for kNN, or provide radius.")
        k_use = int(k)
    else:
        if radius <= 0:
            raise ValueError("radius must be positive.")

    if indices is None:
        eval_idx = np.arange(N, dtype=int)
    else:
        eval_idx = np.asarray(indices, dtype=int).ravel()

    tree = cKDTree(P)

    plane_rms = np.full((N,), np.nan, dtype=np.float64)
    curvature = np.full((N,), np.nan, dtype=np.float64)
    normal_var = None if N_in is None else np.full((N,), np.nan, dtype=np.float64)
    n_fit = np.full((N, 3), np.nan, dtype=np.float64)
    neighbor_count = np.zeros((N,), dtype=np.int32)

    for i in eval_idx:
        if radius is None:
            _, nn = tree.query(P[i], k=k_use)
            nn = np.asarray(nn, dtype=int)
        else:
            nn = np.asarray(tree.query_ball_point(P[i], r=radius), dtype=int)

        neighbor_count[i] = nn.size
        if nn.size < min_neighbors:
            continue

        Q = P[nn]
        mu = Q.mean(axis=0, keepdims=True)
        X = Q - mu

        C = (X.T @ X) / max(X.shape[0], 1)
        w, V = np.linalg.eigh(C)  # ascending
        w = np.maximum(w, 0.0)

        n = V[:, 0]
        n = n / max(np.linalg.norm(n), eps)
        n_fit[i] = n

        proj = (X @ n)
        plane_rms[i] = float(np.sqrt(np.mean(proj * proj)))

        tr = float(w.sum())
        curvature[i] = float(w[0] / max(tr, eps))

        if N_in is not None:
            ni = N_in[i]
            Nj = N_in[nn]
            dots = Nj @ ni
            if normal_var_use_abs_dot:
                dots = np.abs(dots)
            dots = np.clip(dots, -1.0, 1.0)
            normal_var[i] = float(np.mean(1.0 - dots))

    summary = {}
    summary.update(summarize_1d(plane_rms[eval_idx], "plane_rms"))
    summary.update(summarize_1d(curvature[eval_idx], "curvature"))
    if normal_var is not None:
        summary.update(summarize_1d(normal_var[eval_idx], "normal_var"))

    return SurfaceMetrics(
        plane_rms=plane_rms,
        curvature=curvature,
        normal_var=normal_var,
        n_fit=n_fit,
        neighbor_count=neighbor_count,
        summary=summary,
    )
