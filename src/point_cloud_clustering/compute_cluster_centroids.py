import numpy as np
from typing import Tuple
from .KMeansClusterer import KMeansClusterer


def compute_cluster_centroids(
    points: np.ndarray,
    n_clusters: int,
    max_iter: int = 100,
    tol: float = 1e-4,
    random_state: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    点群を n_clusters 個にクラスタリングし，各クラスタの重心位置を返すユーティリティ関数。

    Parameters
    ----------
    points : np.ndarray, shape (N_points, D)
        オブジェクト全体の点群（D=3 を想定）
    n_clusters : int
        生成するクラスタ数
    max_iter : int, default=100
        K-means の最大反復回数
    tol : float, default=1e-4
        収束判定のしきい値
    random_state : int or None, default=None
        乱数シード

    Returns
    -------
    centers : np.ndarray, shape (n_clusters, D)
        各クラスタの重心位置（grasp の初期並進候補として利用）
    labels : np.ndarray, shape (N_points,)
        各点がどのクラスタに属したかのインデックス
    """
    km = KMeansClusterer(
        n_clusters=n_clusters,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
    ).fit(points)
    return km.cluster_centers_, km.labels_

