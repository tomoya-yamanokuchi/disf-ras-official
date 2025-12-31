import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def set_grid_line_3d(
        ax       : Axes3D,
        color    : str = "lightgray",
        linestyle: str = "-",
        linewidth: float = 0.2,
    ):
    # グリッド設定 (Axes3D 独自の設定)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"]['color']     = color       # グリッド線の色
        axis._axinfo["grid"]['linestyle'] = linestyle   # グリッド線のスタイル
        axis._axinfo["grid"]['linewidth'] = linewidth   # グリッド線の太さ
