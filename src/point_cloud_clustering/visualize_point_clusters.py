import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.lines import Line2D
from service import set_grid_line_3d


def _set_axes_equal_from_points(ax, points, centers=None):
    """points/centers が占める範囲から x,y,z を等スケールにそろえる。"""
    pts = np.asarray(points)
    if centers is not None:
        pts = np.vstack([pts, np.asarray(centers)])

    x_min, y_min, z_min = pts.min(axis=0)
    x_max, y_max, z_max = pts.max(axis=0)

    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)
    z_mid = 0.5 * (z_min + z_max)

    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) * 0.5

    ax.set_xlim(x_mid - max_range, x_mid + max_range)
    ax.set_ylim(y_mid - max_range, y_mid + max_range)
    ax.set_zlim(z_mid - max_range, z_mid + max_range)



# elev =  200 #
# azim = -30 #


elev =  200 #
azim = -190 #

def visualize_point_clusters(points,
                            labels,
                            centers=None,
                            point_size=1,
                            annotate_centers=True,
                            show_axis_numbers=True,
                            save     : bool = False,
                            save_path: str = None,
                            figsize=(5,5),
                            ):

    points = np.asarray(points)
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)

    cmap = plt.get_cmap("tab20")
    n_colors = cmap.N
    cluster_colors = {cid: cmap(int(cid) % n_colors) for cid in unique_labels}

    fig = plt.figure(figsize=figsize)

    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=elev, azim=azim)

    # --- クラスタごとに点群描画 ---
    for cid in unique_labels:
        mask = labels == cid
        pts_c = points[mask]
        color = cluster_colors[cid]
        ax.scatter(
            pts_c[:, 0], pts_c[:, 1], pts_c[:, 2],
            s=point_size,
            c=[color],
            label=f"cluster {cid}",
            depthshade=True,
        )

    # --- 重心描画 ---
    if centers is not None:
        centers = np.asarray(centers)
        for cid in unique_labels:
            center = centers[cid]
            color = cluster_colors[cid]
            ax.scatter(
                center[0], center[1], center[2],
                s=60,
                c=[color],
                marker="x",
                linewidths=2,
            )
            if annotate_centers:
                ax.text(
                    center[0], center[1], center[2],
                    f"{cid}",
                    fontsize=10,
                    color="k",
                    ha="center",
                    va="bottom",
                )

    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_zlabel("Z")
    # ax.set_title("Point cloud clustered by k-means")
    # ax.set_title("Point cloud clustered by k-means")

    # 凡例
    legend_elements = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=cluster_colors[cid],
               markersize=8, label=f"cluster {cid}")
        for cid in unique_labels
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    # ★ ここで等スケールにする
    _set_axes_equal_from_points(ax, points, centers=centers)
    set_grid_line_3d(ax)

    # ★ ここで軸の数字だけ消す（グリッドは残す）
    if not show_axis_numbers:
        # tick の位置はそのままにして、ラベルだけ空にする
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        # 念のため：全方向のラベル表示をオフ
        ax.tick_params(labelbottom=False, labelleft=False,
                       labelright=False, labeltop=False)


    if save == False:
        plt.show()

    elif save == True:
        plt.savefig(
            save_path,
            dpi=500,
            bbox_inches='tight'
        )
        plt.close()

    print(" save == ", save)
