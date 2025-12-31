import numpy as np


def print_cluster_centers_pretty(centers: np.ndarray, precision: int = 6):
    """
    K-means のクラスタ中心を，index 付き & カンマ区切りで
    config 用にコピペしやすい形で print する。

    出力イメージ:
      # cluster 0
      [0.061768, 0.104839, -0.054039],
      # cluster 1
      [0.042450, -0.019043, -0.007850],
      ...

    Parameters
    ----------
    centers : (K, 3) ndarray
        クラスタ中心
    precision : int
        小数点以下の桁数
    """
    centers = np.asarray(centers)
    fmt = f"{{:.{precision}f}}"

    print("object_kmeans_centers (for config):")
    for idx, c in enumerate(centers):
        numbers = ", ".join(fmt.format(v) for v in c)
        print(f"  # cluster {idx} ")
        print(f"  [{numbers}]")
        print(f"\n")
