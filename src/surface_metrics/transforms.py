from __future__ import annotations
import numpy as np

Array = np.ndarray


def transform_points_W_to_G(points_W: Array, R_WG: Array, t_WG: Array) -> Array:
    """
    World(W) -> Gripper(G)
    R_WG: (3,3) rotation from G to W
    t_WG: (3,)  origin of G in W
    """
    points_W = np.asarray(points_W, dtype=np.float64)
    R_WG     = np.asarray(R_WG, dtype=np.float64)
    t_WG     = np.asarray(t_WG, dtype=np.float64).reshape(3)
    R_GW     = R_WG.T
    return (points_W - t_WG[None, :]) @ R_GW.T
