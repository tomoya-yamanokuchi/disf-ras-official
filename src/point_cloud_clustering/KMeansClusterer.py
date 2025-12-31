import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class KMeansClusterer:
    """
    Object surface point cloud を N 個にクラスタリングし，
    各クラスタの重心位置を grasp proposal の初期並進として利用するための簡易 K-means 実装。

    Parameters
    ----------
    n_clusters : int
        生成するクラスタ数（= 初期並進候補の数）
    max_iter : int, default=100
        K-means の最大反復回数
    tol : float, default=1e-4
        収束判定用の重心移動量のしきい値（L2 ノルム）
    random_state : int or None, default=None
        乱数シード（再現性のために固定したい場合に指定）
    """
    n_clusters: int
    max_iter: int = 100
    tol: float = 1e-4
    random_state: int | None = None

    # 学習後に埋まるフィールド
    cluster_centers_: np.ndarray | None = None  # (n_clusters, 3)
    labels_: np.ndarray | None = None           # (N_points,)

    def fit(self, points: np.ndarray) -> "KMeansClusterer":
        """
        点群に対して K-means を適用し，クラスタ重心とラベルを求める。

        Parameters
        ----------
        points : np.ndarray, shape (N_points, D)
            オブジェクト全体の点群（D=3 を想定）

        Returns
        -------
        self : KMeansClusterer
            学習済みインスタンス（cluster_centers_, labels_ がセットされる）
        """
        points = np.asarray(points, dtype=float)
        if points.ndim != 2:
            raise ValueError("points must be 2D array with shape (N_points, D).")
        n_points, dim = points.shape
        if self.n_clusters > n_points:
            raise ValueError("n_clusters must be <= number of points.")

        rng = np.random.default_rng(self.random_state)

        # --- 初期重心：点群からランダムに n_clusters 個サンプル ---
        init_indices = rng.choice(n_points, size=self.n_clusters, replace=False)
        centers = points[init_indices].copy()

        for _ in range(self.max_iter):
            # 各点から各クラスタ中心への距離（二乗）を計算
            diff = points[:, None, :] - centers[None, :, :]  # (N_points, n_clusters, D)
            dists_sq = np.sum(diff ** 2, axis=2)             # (N_points, n_clusters)

            # 最も近いクラスタに割り当て（ラベル）
            labels = np.argmin(dists_sq, axis=1)             # (N_points,)

            # 新しいクラスタ中心を計算
            new_centers = np.zeros_like(centers)
            for k in range(self.n_clusters):
                mask = labels == k
                if np.any(mask):
                    new_centers[k] = points[mask].mean(axis=0)
                else:
                    # 空クラスタが出た場合はランダムな点で再初期化
                    new_centers[k] = points[rng.integers(0, n_points)]

            # 収束判定：重心の最大移動量が tol 未満なら終了
            shift = np.linalg.norm(new_centers - centers, axis=1).max()
            centers = new_centers
            if shift < self.tol:
                break

        self.cluster_centers_ = centers
        self.labels_ = labels
        return self


