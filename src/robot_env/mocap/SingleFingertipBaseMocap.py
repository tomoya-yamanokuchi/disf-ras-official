import numpy as np
from service import transform
from service import generate_surface_points_panda_hand
from service import generate_surface_normals_panda_hand
from service import unbias_fingertip_surface_geom_pos_offset
from ..utils.GeomManager import GeomManager
from ..utils.BodyManager import BodyManager
from .MocapManager import MocapManager
from .SingleFingertipBaseMocapInstanceDict import SingleFingertipBaseMocapInstanceDict


class SingleFingertipBaseMocap:
    def __init__(self, paramsDict: SingleFingertipBaseMocapInstanceDict):
        self.data                         = paramsDict['data']
        self.geom          : GeomManager  = paramsDict['geom']
        self.body          : BodyManager  = paramsDict['body']
        self.mocap_manager : MocapManager = paramsDict['mocap_manager']
        self.mocap_name                   = paramsDict['mocap_name']
        self.normal_gripper               = np.array(paramsDict['normal_gripper'])
        self.scale_param                  = paramsDict['scale_param']
        self.name_body_hand               = paramsDict['name_body_hand']
        self.name_geom_fingertip          = paramsDict['name_geom_fingertip']
        self.id_body_hand                 = self.body.name2id(self.name_body_hand)
        self.id_geom_fingertip            = self.geom.name2id(self.name_geom_fingertip)
        self.points_local                 = generate_surface_points_panda_hand(self.name_geom_fingertip)
        self.normal_local                 = generate_surface_normals_panda_hand(
            self.name_geom_fingertip, self.normal_gripper, num_points=self.points_local.shape[0]
        )
        # ---
        self.rotation_matrix    = None
        self.translation_vector = None
        self.points_world_start = None
        self.points_world_end   = None
        self.normal_world       = None

    def update(self):
        self.__update_posture()
        self.__update_position_and_normal()
        self.__update_mocap()

    def __update_posture(self):
        self.rotation_matrix    = self.body.xmat(body_id=self.id_body_hand)      # (3, 3)
        geom_pos                = self.geom.xpos(geom_id=self.id_geom_fingertip) # (3,)
        self.translation_vector = unbias_fingertip_surface_geom_pos_offset(geom_pos, self.name_geom_fingertip)

    def __update_position_and_normal(self):
        self.points_world_start = transform(self.points_local, self.rotation_matrix, self.translation_vector)
        self.normal_world       = np.dot(self.normal_local, self.rotation_matrix.T)
        self.points_world_end   = self.points_world_start + (self.normal_world * self.scale_param)
        # import ipdb ; ipdb.set_trace()

    def __update_mocap(self):
        for i in range(len(self.points_world_start)):
            self.mocap_manager.set_pos(mocap_body_id=self.mocap_manager.name2id(f"{self.mocap_name}_start{i}"), mocap_pos=self.points_world_start[i])
            self.mocap_manager.set_pos(mocap_body_id=self.mocap_manager.name2id(f"{self.mocap_name}_end{i}"),  mocap_pos=self.points_world_end[i])

    def get_points_world(self):
        return self.points_world_start

