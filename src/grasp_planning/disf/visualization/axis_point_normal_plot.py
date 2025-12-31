import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from value_object import PointNormalUnitPairs


def axis_point_normal_plot(
        ax           : Axes3D,
        point_normal : PointNormalUnitPairs,
        label        : str,
        color        : str,
        # ----
        point_size   : float = 10,
        normal_length: float = 0.005,
        point_alpha  : float = 1.0,
        normal_alpha : float = 1.0,
    ):
    # -----------------
    p   = point_normal.points
    n_p = point_normal.normals
    # ----- point -----
    ax.scatter(
        p[:, 0], p[:, 1], p[:, 2],
        # ---
        c     = color,
        label = label,
        s     = point_size,
        alpha = point_alpha,
    )
    # ----- normal -----
    ax.quiver(
          p[:, 0],   p[:, 1],   p[:, 2],
        n_p[:, 0], n_p[:, 1], n_p[:, 2],
        # ---
        color  = color,
        length = normal_length,
        alpha  = normal_alpha,
    )
