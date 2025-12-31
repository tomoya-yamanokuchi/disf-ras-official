from .HandBaseEnv import DexterousHandEnv
from domain_object.builder import DomainObject
import numpy as np
# from .pose import PandaHandPose


class PandaHandSimpleEnv(DexterousHandEnv):
    def __init__(self, domain_object: DomainObject):
        super().__init__(domain_object)
        # self.pose = PandaHandPose(domain_object)

    def set_params(self):
        self.__set_ids()

    def __set_ids(self):
        self.id_right_fingertip = self.geom.name2id("right_fingertip_center")
        self.id_left_fingertip  = self.geom.name2id("left_fingertip_center")
        self.id_object          = self.geom.name2id("object")

    def set_qpos_object(self, qpos_object: np.ndarray):
        assert qpos_object.shape == (7,) # pos + quat
        self.data.qpos[-7:] = qpos_object

    def set_qpos_palm(self, qpos_palm: np.ndarray):
        assert qpos_palm.shape == (7,) # pos + quat
        self.data.qpos[:7] = qpos_palm

    def set_qpos_finger(self, qpos_finger: np.ndarray):
        self.data.qpos[7:7+2] = qpos_finger

    def set_ctrl_finger(self, ctrl_finger: np.ndarray):
        self.data.ctrl[:] = ctrl_finger
        # import ipdb ; ipdb.set_trace()
