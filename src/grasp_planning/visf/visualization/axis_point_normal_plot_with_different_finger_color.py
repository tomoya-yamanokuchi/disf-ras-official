import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from value_object import PointNormalIndexUnitPairs


def axis_point_normal_plot_with_different_finger_color(
        ax               : Axes3D,
        point_normal     : PointNormalIndexUnitPairs,
        label            : str,
        finger_color_dict: dict,
        point_size       : float = 10,
        normal_length    : float = 0.005,
        point_alpha      : float = 1.0,
        normal_alpha     : float = 1.0,

    ):
    # ----
    for key, val in finger_color_dict.items():
        mask = point_normal.finger_indices == int(key)
        p    = point_normal.points[mask]
        n_p  = point_normal.normals[mask]

        # ----- point -----
        ax.scatter(
            p[:, 0], p[:, 1], p[:, 2],
            # ---
            c     = val,
            label = label,
            s     = point_size,
            alpha = point_alpha,
        )
        # ----- normal -----
        ax.quiver(
            p[:, 0],   p[:, 1],   p[:, 2],
            n_p[:, 0], n_p[:, 1], n_p[:, 2],
            # ---
            color  = val,
            length = normal_length,
            alpha  = normal_alpha,
        )
