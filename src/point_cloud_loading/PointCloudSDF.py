# PointCloudSDF.py など
import numpy as np
from scipy.spatial import cKDTree

class PointCloudSDF:
    def __init__(self, point_cloud):
        self.point_cloud    = point_cloud
        self.sdf            = None
        self.sdf_grid_min   = None
        self.sdf_voxel_size = None
        self.sdf_grid_shape = None

    def build_ycb_sdf(self, voxel_size: float = 0.004, padding: float = 0.02):
        pts = np.asarray(self.point_cloud.points)   # (N, 3)

        bbox_min = pts.min(axis=0) - padding
        bbox_max = pts.max(axis=0) + padding

        extent = bbox_max - bbox_min
        grid_shape = np.ceil(extent / voxel_size).astype(int)
        Nx, Ny, Nz = grid_shape
        print(f"SDF grid shape: {Nx} x {Ny} x {Nz}")

        xs = bbox_min[0] + (np.arange(Nx) + 0.5) * voxel_size
        ys = bbox_min[1] + (np.arange(Ny) + 0.5) * voxel_size
        zs = bbox_min[2] + (np.arange(Nz) + 0.5) * voxel_size

        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        grid_points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

        tree = cKDTree(pts)
        # ★ここを workers に
        dists, _ = tree.query(grid_points, k=1, workers=-1)

        sdf = dists.reshape(Nx, Ny, Nz).astype(np.float32)

        self.sdf            = sdf
        self.sdf_grid_min   = bbox_min
        self.sdf_voxel_size = float(voxel_size)
        self.sdf_grid_shape = grid_shape

        # ついでに重心もメモ
        self.object_centroid = pts.mean(axis=0)



    # --- SDF サンプリング ---
    def sample_sdf_nearest(self, xyz: np.ndarray) -> np.ndarray:
        """
        xyz: (..., 3)
        return: 同じshapeのスカラー配列 (...,)
        """
        xyz = np.asarray(xyz)

        # grid の index 空間に変換
        rel = (xyz - self.sdf_grid_min) / self.sdf_voxel_size   # (..., 3)

        idx = np.rint(rel).astype(int)   # 最近ボクセル
        Nx, Ny, Nz = self.sdf_grid_shape

        # 範囲外をクリップ
        idx[..., 0] = np.clip(idx[..., 0], 0, Nx-1)
        idx[..., 1] = np.clip(idx[..., 1], 0, Ny-1)
        idx[..., 2] = np.clip(idx[..., 2], 0, Nz-1)

        d = self.sdf[idx[..., 0], idx[..., 1], idx[..., 2]]
        return d



    def visualize_sdf_slice(self, axis: str = "z", index: int | None = None):
        import matplotlib.pyplot as plt
        import numpy as np

        """
        SDF の1スライスを matplotlib で表示。
        axis: "x" / "y" / "z" のどれか
        index: スライス番号（None のときは中央）
        """
        sdf        = self.sdf
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


    def score_hand_pose_by_sdf(self, R: np.ndarray, t: np.ndarray,
                            band: float = 0.005) -> float:
        """
        与えられた hand pose (R,t) に対して、
        canonical fingertip surface self.source.points を SDF 上で評価し、
        「表面に近い点が多いほど高スコア」となる値を返す。
        """
        # 1) ローカル → world
        F_hand = self.source.points          # (M,3)
        P_world = transform_points(R, t, F_hand)  # (M,3)

        # 2) SDF サンプリング
        d = self.sample_sdf_nearest(P_world)     # (M,)

        # 3) band reward
        val = rho_band(d, band=band)            # (M,)
        score = float(np.sum(val))

        # optional: デバッグのために統計も一緒に返してもいい
        # return score, d.min(), d.mean()

        return score
