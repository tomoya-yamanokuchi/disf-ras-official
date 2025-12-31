import numpy as np
from service import transform_gripper, direction_index
from .SingleFingertipTransformedMocapInstanceDict import SingleFingertipTransformedMocapInstanceDict
from .FingertipTransformParamsDict import FingertipTransformParamsDict
from .MocapManager import MocapManager
from value_object import GripperTransformationParams


class SingleFingertipTransformedMocap:
    def __init__(self, paramsDict: SingleFingertipTransformedMocapInstanceDict):
        self.mocap_manager : MocapManager = paramsDict['mocap_manager']
        self.fingertip_base_mocap         = paramsDict['fingertip_base_mocap']
        self.mocap_name                   = paramsDict['mocap_name']
        self.scale_param                  = paramsDict['scale_param']
        # ---
        self.j = direction_index(self.fingertip_base_mocap.name_geom_fingertip)
        self.v = self.fingertip_base_mocap.normal_gripper
        # ---
        self.normal_world       = None
        self.points_world_start = None
        self.points_world_end   = None

    # -----------------------------------------------------------------------------------
    def update_transform_into_base(self, gripper_transform_params: GripperTransformationParams):
        self.__update_transform_into_base(gripper_transform_params)
        self.__update_mocap()

    def __update_transform_into_base(self, gripper_transform_params: GripperTransformationParams):
        self.normal_world       = np.dot(self.fingertip_base_mocap.normal_world, gripper_transform_params['rotation_matrix'].T)
        self.points_world_start = transform_gripper(points=self.fingertip_base_mocap.points_world_start, j=self.j, **gripper_transform_params)
        self.points_world_end   = self.points_world_start + (self.fingertip_base_mocap.normal_world * self.scale_param)
        # import ipdb ; ipdb.set_trace()

    # -----------------------------------------------------------------------------------
    def update_transform_into_self(self, gripper_transform_params: GripperTransformationParams):
        self.__update_transform_into_self(gripper_transform_params)
        self.__update_mocap()

    def __update_transform_into_self(self, gripper_transform_params: GripperTransformationParams):
        self.normal_world       = np.dot(self.normal_world, gripper_transform_params['rotation_matrix'].T)
        self.points_world_start = transform_gripper(points=self.points_world_start, j=self.j, **gripper_transform_params)
        self.points_world_end   = self.points_world_start + (self.normal_world * self.scale_param)

    # =======================
    def __update_mocap(self):
        for i in range(len(self.points_world_start)):
            self.mocap_manager.set_pos(mocap_body_id=self.mocap_manager.name2id(f"{self.mocap_name}_start{i}"),  mocap_pos=self.points_world_start[i])
            self.mocap_manager.set_pos(mocap_body_id=self.mocap_manager.name2id(f"{self.mocap_name}_end{i}"),    mocap_pos=self.points_world_end[i])

    def get_points_world(self):
        return self.points_world_start

    def get_normals_world(self):
        return self.normal_world

