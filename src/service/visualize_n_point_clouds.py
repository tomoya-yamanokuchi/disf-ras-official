import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from typing import List
from value_object import PointNormalUnitPairs
from .set_aspect_equal_3d import set_aspect_equal_3d


def visualize_n_point_clouds(
        point_normals: List[PointNormalUnitPairs],
        labels       : list = None,
        colors       : list = None,
        title        : str = "",
        save_path    : str = "./visualize_n_point_clouds.png",
        elev         : int = 20,
        azim         : int = 5,
        normal_length: float = 0.005, # 0.025,
        min_range    : float = 0.03, # 最小の共通レンジの値を指定
        gamma        : float = 1,    # z軸のスケールパラメータ
        mode         : int = 0,
        legend_location = "lower right",
        label_fontsize  = 14,
        tick_fontsize   = 12,
        figsize = (6,6),
        point_size : float = 1.5,
        xlim : tuple = (None, None),
        ylim : tuple = (None, None),
        zlim : tuple = (None, None),
    ):
    # -------------------------------------------
    # 3Dプロットを作成
    fig = plt.figure(figsize=figsize)
    ax : Axes3D = fig.add_subplot(111, projection='3d')

    # デフォルトの色リスト
    default_colors = ['r', 'b', 'g', 'y', 'm', 'c', 'k']

    # 各点群をプロット
    point_clouds = []
    all_points = []  # 全てのポイントを集めておくリスト
    for i, point_normal in enumerate(point_normals):

        point_cloud = point_normal.points
        point_clouds.append(point_cloud)
        all_points.append(point_cloud)  # 各点群をリストに追加
        normal_vectors = point_normal.normals

        # 色とラベルを決定
        color = colors[i] if colors and i < len(colors) else default_colors[i % len(default_colors)]
        label = labels[i] if labels and i < len(labels) else f'Point Cloud {i + 1}'

        # 点群をプロット
        ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c=color, label=label, s=point_size)

        # 法線ベクトルをプロット
        ax.quiver(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2],
                  normal_vectors[:, 0], normal_vectors[:, 1], normal_vectors[:, 2],
                  length=normal_length, color=color, alpha=0.5)

    # 全てのポイントクラウドの最小値・最大値を求めて共通の軸範囲を決定
    all_points = np.concatenate(all_points, axis=0)  # 全てのポイントを一つの配列にまとめる
    x_min, y_min, z_min = np.min(all_points, axis=0)
    x_max, y_max, z_max = np.max(all_points, axis=0)

    # 全ての軸で共通のレンジを計算し、min_rangeでクリップ
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2.0
    max_range = np.clip(max_range, min_range, None)  # min_range以上になるようにクリップ

    # # 各軸の中心
    # mid_x = (x_max + x_min) / 2.0
    # mid_y = (y_max + y_min) / 2.0
    # mid_z = (z_max + z_min) / 2.0

    # 各軸の範囲を共通に設定。ただしz軸はgamma倍する
    # ax.set_xlim(mid_x - max_range, mid_x + max_range)
    # ax.set_ylim(mid_y - max_range, mid_y + max_range)
    # ax.set_zlim(mid_z - gamma * max_range, mid_z + gamma * max_range)

    # ax.set_xlim(*xlim)
    # ax.set_ylim(*ylim)
    # ax.set_zlim(zlim[0]*gamma, zlim[1]*gamma)

    # ラベルとタイトルを追加
    ax.set_xlabel('X', fontsize=label_fontsize)
    ax.set_ylabel('Y', fontsize=label_fontsize)
    ax.set_zlabel('Z', fontsize=label_fontsize)

    ax.set_title(title)

    plt.title(title, fontsize=label_fontsize)
    # import ipdb ; ipdb.set_trace()

    # 凡例を表示
    # ax.legend()
    # plt.legend(loc=legend_location, fontsize=tick_fontsize)  # Set legend location

    set_aspect_equal_3d(ax)

    # カメラの視点を調整
    ax.view_init(elev=elev, azim=azim)

    # plt.show()
    # # プロットを表示
    if mode == 0:
        plt.show()
        plt.close()
    elif mode == 1:
        plt.tight_layout()
        plt.savefig(save_path, dpi=500, bbox_inches='tight')
        plt.close()

# PointNormalUnitPairsのサンプルデータを用意して関数を呼び出す
