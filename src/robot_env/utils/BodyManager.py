import mujoco
from service import ExtendedRotation


class BodyManager:
    def __init__(self, model, data):
        self.model  = model
        self.data   = data

    def id2name(self, id: int):
        return mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, id)

    def name2id(self, body_name: str):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)

    def pos(self, body_id: int):
        pos = self.model.body_pos[body_id]
        return pos

    def quat(self, body_id: int):
        quat = self.model.body_quat[body_id]
        return quat

    def mat(self, body_id: int, reshape: bool=True):
        quat     = self.quat(body_id)
        rotation = ExtendedRotation.from_quat(quat)
        R        = rotation.as_rodrigues()
        if reshape: return R
        else      : return R.flat()

    def xmat(self, body_id: int, reshape: bool=True):
        if reshape: return self.data.xmat[body_id].reshape(3, 3)
        else      : return self.data.xmat[body_id]

    def xpos(self, body_id: int):
        return self.data.xpos[body_id]

    def xquat(self, body_id: int):
        R        = self.xmat(body_id, reshape=True)
        rotation = ExtendedRotation.from_matrix(R)
        quat     = rotation.as_quat_scalar_first()
        return quat
