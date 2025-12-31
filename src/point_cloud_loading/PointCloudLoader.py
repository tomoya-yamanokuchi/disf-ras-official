import numpy as np
import open3d as o3d
from domain_object.builder import DomainObject
from service import ExtendedRotation
from .frame_transformation import frame_transformation


class PointCloudLoader:
    def __init__(self, domain_object: DomainObject):
        cwd_path                         = domain_object.cwd_path
        pc_data_dir_path_relative_to_cwd = domain_object.pc_data_dir_path_relative_to_cwd
        self.model_name                  = domain_object.model_name
        suffix_fname                     = domain_object.suffix_fname
        self.config_pc_data              = domain_object.config_pc_data
        self.rotvec_object_in_pcd_load       = domain_object.rotvec_object_in_pcd_load
        self.translation_object_in_pcd_load  = domain_object.translation_object_in_pcd_load
        assert self.translation_object_in_pcd_load is not None

        # ----------- set file_path -----------
        self.file_path = cwd_path.joinpath(
            pc_data_dir_path_relative_to_cwd,
            self.model_name,
            suffix_fname
        )
        # ----------------------------------------
        self.R_GO = ExtendedRotation.from_rotvec(self.rotvec_object_in_pcd_load).as_rodrigues()
        self.t_GO = self.translation_object_in_pcd_load

    def load(self):
        pcd = o3d.io.read_point_cloud(str(self.file_path))
        self.pcd = pcd.voxel_down_sample(
            voxel_size=self.config_pc_data.pre_prosessing.voxel_down_sample_size
        )

        # import ipdb; ipdb.set_trace()


    def get_point_cloud(self):

        pcd  = self.pcd
        R_GO = self.R_GO
        t_GO = np.zeros(3)

        # --- apply transform to points
        pts             = np.asarray(pcd.points)               # (N,3) in O-frame
        pts_G           = (R_GO @ pts.T).T + t_GO[None, :]     # (N,3) in G-frame
        pcd.points = o3d.utility.Vector3dVector(pts_G)

        # --- 法線もあるなら回転だけ掛け直す
        if pcd.has_normals():
            n = np.asarray(pcd.normals)
            n_G = (R_GO @ n.T).T
            pcd.normals = o3d.utility.Vector3dVector(n_G)

        # --------------- YCB specific ---------------
        z_offset = np.array(self.pcd.points)[:, -1].max()
        translation_correction = np.array([0.0, 0.0, -z_offset])
        pcd.translate(translation_correction)

        transformed_pcd = pcd
        return transformed_pcd


if __name__ == '__main__':
    pcd_loader = PointCloudLoader(domain_object)
    pcd_loader.load()
    pcd_loader.frame_transformation()
    pcd = pcd_loader.get_point_cloud()

