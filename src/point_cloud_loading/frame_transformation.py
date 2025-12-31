import numpy as np
import open3d as o3d


def frame_transformation(pcd, R_GO, t_GO):
    '''
        transform "object frame" to "gripper canonical surface frame"
    '''
    # --- apply transform to points
    pts             = np.asarray(pcd.points)               # (N,3) in O-frame
    pts_G           = (R_GO @ pts.T).T + t_GO[None, :]     # (N,3) in G-frame
    pcd.points = o3d.utility.Vector3dVector(pts_G)

    # --- 法線もあるなら回転だけ掛け直す
    if pcd.has_normals():
        n = np.asarray(pcd.normals)
        n_G = (R_GO @ n.T).T
        pcd.normals = o3d.utility.Vector3dVector(n_G)

    return pcd
