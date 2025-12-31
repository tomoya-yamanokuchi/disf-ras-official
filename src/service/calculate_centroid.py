import numpy as np


def calculate_centroid(point_clouds: np.ndarray, keepdims:bool=True):
    """
    Parameters:
        point_cloud (numpy.ndarray): shape=(N, 3)
    Returns:
        numpy.ndarray: centroid of point_cloud (x, y, z)
    """
    assert (point_clouds.ndim == 2) and (point_clouds.shape[-1] == 3)
    # ---
    centroid = np.mean(point_clouds, axis=0, keepdims=keepdims)
    return centroid # (1, 3) if keepdims is True


if __name__ == '__main__':
    # 使用例
    point_cloud = np.array([[1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0],
                            [7.0, 8.0, 9.0],
                            [10.0, 11.0, 12.0]])

    centroid = calculate_centroid(point_cloud)
    print("点群の重心:", centroid)
