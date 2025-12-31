from .HandBaseEnv import HandBaseEnv
from domain_object.builder import DomainObject
from value_object import GripperTransformationParams


class HandDemoEnv(HandBaseEnv):
    def __init__(self, domain_object: DomainObject):
        super().__init__(domain_object)
        self.config_env         = domain_object.config_env
        self.reaching_threshold = domain_object.config_env.reaching_threshold
        self.finger_reach       = False
        # ----
        self._finger_velocity_buffer = []
        self._finger_grasped         = False
        self.set_params()

    def set_params(self):
        self.__set_ids()

    def __set_ids(self):
        self.id_right_fingertip = self.geom.name2id("right_fingertip_center")
        self.id_left_fingertip  = self.geom.name2id("left_fingertip_center")
        self.id_object          = self.geom.name2id("object")
