import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from typing import List
from value_object import SourcePointSurfaceSet, TargetPointSurfaceSet
# from value_object import PointNormalUnitPairs
from ....service.set_aspect_equal_3d import set_aspect_equal_3d


from .axis_point_normal_plot import axis_point_normal_plot


def visualize_disf(
        source_set: SourcePointSurfaceSet,
        target_set: TargetPointSurfaceSet,
        # ----
        title        : str = "",
        save_path    : str = "./visualize_n_point_clouds_disf.png",
        elev         : int = 20,
        azim         : int = 5,
        normal_length: float = 0.005, # 0.025,
        min_range    : float = 0.03, # 最小の共通レンジの値を指定
        gamma        : float = 1,    # z軸のスケールパラメータ
        mode         : int = 0,
        legend_location = "lower right",
        label_fontsize  = 14,
        tick_fontsize   = 12,
        figsize : tuple = (6, 6),
        point_size : float = 1.5,
        xlim : tuple = (None, None),
        ylim : tuple = (None, None),
        zlim : tuple = (None, None),
    ):
    # -------------------------------------------
    fig         = plt.figure(figsize=figsize)
    ax : Axes3D = fig.add_subplot(111, projection='3d')

    # デフォルトの色リスト
    # default_colors = ['r', 'b', 'g', 'y', 'm', 'c', 'k']
    colors                = [
        'r', 'plum',
        'b', 'skyblue', 'gray',
    ],



    import ipdb ; ipdb.set_trace()

    # 各点群をプロット
    point_clouds = []
    all_points = []  # 全てのポイントを集めておくリスト
    for i, point_normal in enumerate(point_normals):
        point_cloud = point_normal.points
        point_clouds.append(point_cloud)
        all_points.append(point_cloud)  # 各点群をリストに追加
        normal_vectors = point_normal.normals

        # 色とラベルを決定
        color = colors[i]
        label = labels[i]

        # 点群をプロット
        ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2],
            c=color, label=label,
            s=point_size*5)

        # 法線ベクトルをプロット
        ax.quiver(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2],
                  normal_vectors[:, 0], normal_vectors[:, 1], normal_vectors[:, 2],
                  length=normal_length*2,
                  color=color,
                  alpha=1)



    # ----------------------- surface -----------------------
    for i, point_normal in enumerate(surface_point_normals):
        point_cloud = point_normal.points
        point_clouds.append(point_cloud)
        all_points.append(point_cloud)  # 各点群をリストに追加
        normal_vectors = point_normal.normals

        # 色とラベルを決定
        color = surface_colors[i]
        label = surface_labels[i]

        # 点群をプロット
        ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2],
                c=color, label=label, s=point_size)

        # 法線ベクトルをプロット
        ax.quiver(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2],
                  normal_vectors[:, 0], normal_vectors[:, 1], normal_vectors[:, 2],
                  length =normal_length,
                  color=color,
                  alpha=1
        )

