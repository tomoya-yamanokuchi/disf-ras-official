import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# アスペクト比を1:1:1にするための関数
def set_aspect_equal_3d(ax):
    extents = np.array([getattr(ax, f'get_{dim}lim')() for dim in 'xyz'])
    centers = np.mean(extents, axis=1)
    max_size = max(extents[:, 1] - extents[:, 0])
    r = 0.5 * max_size
    # import ipdb; ipdb.set_trace()
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, f'set_{dim}lim')(ctr - r, ctr + r)


if __name__ == '__main__':
    # プロットの設定
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # サンプルデータをプロット
    x = np.linspace(-0.05, 0.05, 100)
    y = np.linspace(-0.05, 0.05, 100)
    z = np.linspace(-0.05, 0.05, 100)
    ax.plot(x, y, z)

    # アスペクト比を設定
    set_aspect_equal_3d(ax)

    # プロットの表示
    plt.show()
