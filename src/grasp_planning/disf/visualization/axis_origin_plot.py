import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from value_object import PointNormalUnitPairs


def axis_origin_plot(
        ax           : Axes3D,
        label        : str,
        color        : str = "red",
        # ----
        point_size   : float = 10,
        point_alpha  : float = 1.0,
    ):
    # ----- point -----
    ax.scatter(
        0,0,0,
        # ---
        c     = color,
        label = label,
        s     = point_size,
        alpha = point_alpha,
    )
    # ax.quiver(
    #     0,0,0,
    #     1,1,1,
    #     # ---
    #     color  = color,
    #     length = 0.1,
    #     alpha  = point_alpha,
    # )

