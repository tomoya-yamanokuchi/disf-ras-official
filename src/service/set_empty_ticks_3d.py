import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def set_empty_ticks_3d(
        ax: Axes3D,
    ):
    # 軸に何も表示しない
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
