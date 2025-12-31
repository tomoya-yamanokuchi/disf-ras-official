import mujoco
import numpy as np

class GeomManager:
    def __init__(self, model, data):
        self.model  = model
        self.data   = data

    def name2id(self, geom_name: str):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)

    def parent_body_id(self, geom_id: int):
        return self.model.geom_bodyid[geom_id]

    def xpos(self, geom_id: int):
        return self.data.geom_xpos[geom_id]

    def xmat(self, geom_id: int, reshape: bool=True):
        if reshape: return self.data.geom_xmat[geom_id].reshape(3, 3)
        else      : return self.data.geom_xmat[geom_id]

    def xquat(self, geom_id: int):
        geom_quat = np.zeros(4)
        mujoco.mju_mat2Quat(geom_quat, self.xmat(geom_id, reshape=False))
        return geom_quat

    def size(self, geom_id: int):
        return self.model.geom_size[geom_id]

    def pos(self, geom_id: int):
        # static position
        return self.model.geom_pos[geom_id]

    def quat(self, geom_id: int):
        # static quatanion
        return self.model.geom_quat[geom_id]

    def mat(self, geom_id: int, reshape: bool=True):
        # static matrix
        geom_mat = np.zeros(9) # initialize for mju_quat2Mat
        mujoco.mju_quat2Mat(geom_mat, self.quat(geom_id))
        if reshape: return geom_mat.reshape(3, 3)
        else      : return geom_mat


    def set_pos(self, geom_id: int, pos: np.ndarray):
        self.data.geom_pos[geom_id][:] = pos[:]

    def set_xpos(self, geom_id: int, pos: np.ndarray):
        self.data.geom_xpos[geom_id][:] = pos[:]

    def set_xmat(self, geom_id: int, rotation_matix: np.ndarray):
        self.data.geom_xmat[geom_id][:] = rotation_matix.flatten()
