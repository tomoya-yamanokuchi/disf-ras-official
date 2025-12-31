import os
import numpy as np
import open3d as o3d


def cutom_point_cloud_load(
        mesh_path       : str,
        number_of_points: int = 3000
    ):

    # import ipdb; ipdb.set_trace()
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if mesh.is_empty():
        raise ValueError(f"Loaded mesh is empty: {mesh_path}")

    pcd = mesh.sample_points_poisson_disk(number_of_points)


    pts = np.asarray(pcd.points)   # 現在の単位(例: mm)
    pts_m = pts * 0.001            # メートルに変換

    if pts_m.size == 0:
        raise ValueError("No points were sampled from the mesh.")

    return pts_m
