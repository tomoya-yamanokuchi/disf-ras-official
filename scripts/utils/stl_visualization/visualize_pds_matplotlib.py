"""Visualize Poisson-disk sampled point clouds from meshes using matplotlib.

Usage:
    python visualize_pds_matplotlib.py --mesh /path/to/mesh.stl --npts 500 --save out.png

If --save is omitted, the matplotlib window will be shown interactively.
"""

import argparse
import os

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def visualize_pds_with_matplotlib(
        point_clouds,
        save_path: str = None, point_size: float = 1.0):


    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    pts_m = point_clouds
    z     = pts_m[:, 2]
    sc    = ax.scatter(pts_m[:, 0], pts_m[:, 1], pts_m[:, 2], c=z, cmap='viridis', s=point_size)

    plt.colorbar(sc, ax=ax, shrink=0.6, label='Z')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Poisson Disk Sampled Point Cloud ({pts_m.shape[0]} pts)')

    try:
        max_range = (pts_m.max(axis=0) - pts_m.min(axis=0)).max() / 2.0
        mid = (pts_m.max(axis=0) + pts_m.min(axis=0)) / 2.0
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    except Exception:
        pass

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()


def _parse_args():
    p = argparse.ArgumentParser(description='Visualize Poisson-disk sampled point cloud from a mesh using matplotlib')
    p.add_argument('--mesh', '-m', type=str, default='/home/cudagl/disf_ras/models/custom_mesh/square.stl', help='Path to mesh file (STL/OBJ)')
    p.add_argument('--npts', '-n', type=int, default=500, help='Number of Poisson-disk sample points')
    p.add_argument('--save', '-s', type=str, default=None, help='If provided, save the figure to this path (PNG)')
    p.add_argument('--size', type=float, default=1.0, help='Point size for scatter')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()


    args.mesh = "/home/cudagl/disf_ras/models/egad_obj/A0.obj"

    args.npts = 3000

    from point_cloud_loading import cutom_point_cloud_load
    point_clouds = cutom_point_cloud_load(args.mesh, number_of_points=args.npts)

    visualize_pds_with_matplotlib(point_clouds, save_path=args.save, point_size=args.size)

    mesh = o3d.io.read_triangle_mesh('models/custom_mesh/square.stl')
    verts = np.asarray(mesh.vertices)
    print('min', verts.min(axis=0), 'max', verts.max(axis=0))
    print('bbox size', verts.max(axis=0)-verts.min(axis=0))
