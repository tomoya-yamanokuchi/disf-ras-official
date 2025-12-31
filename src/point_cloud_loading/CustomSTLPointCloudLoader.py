import os
import numpy as np
import open3d as o3d
from domain_object.builder import DomainObject
from service import ExtendedRotation
from .frame_transformation import frame_transformation

import warnings
warnings.filterwarnings("ignore", message="Module open3d.cuda.pybind not importable")

class CustomSTLPointCloudLoader:
    def __init__(self, domain_object: DomainObject):
        cwd_path                             = domain_object.cwd_path
        pc_data_dir_path_relative_to_cwd     = domain_object.pc_data_dir_path_relative_to_cwd
        model_name                           = domain_object.config_pc_data.model_name
        suffix_fname                         = domain_object.suffix_fname
        self.config_pc_data                  = domain_object.config_pc_data
        self.rotvec_object_in_pcd_load       = domain_object.rotvec_object_in_pcd_load
        self.translation_object_in_pcd_load  = domain_object.translation_object_in_pcd_load

        # ----------- set file_path -----------
        self.mesh_path = cwd_path.joinpath(
            pc_data_dir_path_relative_to_cwd,
            model_name,
            suffix_fname
        )
        # ----------------------------------------
        self.R_GO = ExtendedRotation.from_rotvec(self.rotvec_object_in_pcd_load).as_rodrigues()
        self.t_GO = self.translation_object_in_pcd_load
        # import ipdb; ipdb.set_trace()

    def load(self, number_of_points: int = 3000, seed: int | None = 0):
        if not os.path.exists(self.mesh_path):
            raise FileNotFoundError(f"Mesh file not found: {self.mesh_path}")

        mesh = o3d.io.read_triangle_mesh(self.mesh_path)
        if mesh.is_empty():
            raise ValueError(f"Loaded mesh is empty: {self.mesh_path}")

        # ★ Poisson サンプリングの前に Open3D の乱数シードを固定
        if seed is not None:
            o3d.utility.random.seed(seed)

        self.pcd = mesh.sample_points_poisson_disk(number_of_points)


    def frame_transformation_with_R_and_t(self):
        self.transformed_pcd = frame_transformation(self.pcd, self.R_GO, self.t_GO)

    def frame_transformation_with_z_offset_alignment(self):

        pts   = np.asarray(self.transformed_pcd.points)
        pcd_z = pts[:, -1]
        max_val_pcd_z = max(pcd_z)

        translation_correction = np.array([0.0, 0.0, -max_val_pcd_z])
        pts_G = pts + translation_correction

        self.transformed_pcd.points = o3d.utility.Vector3dVector(pts_G)


    def get_point_cloud(self):
        return self.transformed_pcd


if __name__ == '__main__':
    pcd_loader = PointCloudLoader(domain_object)
    pcd_loader.load()
    pcd_loader.frame_transformation()
    pcd = pcd_loader.get_point_cloud()

