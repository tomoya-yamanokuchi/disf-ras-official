import numpy as np
from scipy.spatial import KDTree
from service import transform
from service import correspondence_filtering
from .BoxObjectMocap import BoxObjectMocap

from .SingleObjectCorrespondenceMocapInstanceDict import SingleObjectCorrespondenceMocapInstanceDict


class SingleObjectCorrespondenceMocap:
    def __init__(self, paramsDict: SingleObjectCorrespondenceMocapInstanceDict):
        self.mocap_manager                 = paramsDict["mocap_manager"]
        self.fingertip_transformed_mocap   = paramsDict["fingertip_transformed_mocap"]
        self.object_mocap : BoxObjectMocap = paramsDict["object_mocap"]
        self.threshold                     = paramsDict["threshold"]
        self.object_mocap_name             = paramsDict["object_mocap_name"]
        self.fingertip_mocap_name          = paramsDict["fingertip_mocap_name"]
        # ---- valid object ---
        self.object_points_world_start    = None
        self.object_points_world_end      = None
        # ---- valid fingertip  ---
        self.fingertip_points_world_start = None
        self.fingertip_points_world_end   = None
        # ---
        self.object_normal_world          = None
        self.fingertip_normal_world       = None
        # ---
        self.max_mocap_num = None

    def update_correspondence_with_single_finger_filtering(self, tree: KDTree):
        distances, indices = tree.query(self.fingertip_transformed_mocap.points_world_start) # compute nearest neighbor
        self.correspondence = correspondence_filtering(
            target_points  = self.object_mocap.points_world_start,
            source_points  = self.fingertip_transformed_mocap.points_world_start,
            distances      = distances,
            indices        = indices,
            threshold      = self.threshold,
        )

    def update_correspondence_with_two_finger_filtering(self, correspondence):
        # --- object (target) ---
        self.object_points_world_start    = correspondence["valid_target"]
        self.valid_target_indices         = correspondence["valid_target_index"]
        self.object_points_world_end      = self.object_mocap.points_world_end[self.valid_target_indices]
        self.object_normal_world          = self.object_mocap.normal_world[self.valid_target_indices]
        # --- fingertip (source) ----
        self.fingertip_points_world_start = correspondence["valid_source"]
        self.valid_source_indices         = correspondence["valid_source_index"]
        self.fingertip_points_world_end   = self.fingertip_transformed_mocap.points_world_end[self.valid_source_indices]
        self.fingertip_normal_world       = self.fingertip_transformed_mocap.get_normals_world()[self.valid_source_indices]
        # ----
        # print(f"valid & invalid idx = [{self.valid_target_indices.shape[0]}, {self.invalid_target_indices.shape[0]}]")
        if self.max_mocap_num is None:
            self.max_mocap_num = self.fingertip_transformed_mocap.points_world_start.shape[0]
        # import ipdb ; ipdb.set_trace()


    def debug_update_correspondence_with_two_finger_filtering(self):
        # --- object (target) ---
        self.object_points_world_start    = self.correspondence["valid_target"]
        self.valid_target_indices         = self.correspondence["valid_target_index"]
        self.object_points_world_end      = self.object_mocap.points_world_end[self.valid_target_indices]
        self.object_normal_world          = self.object_mocap.normal_world[self.valid_target_indices]
        # --- fingertip (source) ----
        self.fingertip_points_world_start = self.correspondence["valid_source"]
        self.valid_source_indices         = self.correspondence["valid_source_index"]
        self.fingertip_points_world_end   = self.fingertip_transformed_mocap.points_world_end[self.valid_source_indices]
        self.fingertip_normal_world       = self.fingertip_transformed_mocap.get_normals_world()[self.valid_source_indices]
        # ----
        # print(f"valid & invalid idx = [{self.valid_target_indices.shape[0]}, {self.invalid_target_indices.shape[0]}]")
        if self.max_mocap_num is None:
            self.max_mocap_num = self.fingertip_transformed_mocap.points_world_start.shape[0]
        # import ipdb ; ipdb.set_trace()


    def update_valid_points_and_normals(self):
        # --- object (target) ---
        self.object_points_world_start    = self.object_mocap.points_world_start[self.valid_target_indices]
        self.object_points_world_end      = self.object_mocap.points_world_end[self.valid_target_indices]
        self.object_normal_world          = self.object_mocap.normal_world[self.valid_target_indices]
        # --- fingertip (source) ----
        self.fingertip_points_world_start = self.fingertip_transformed_mocap.points_world_start[self.valid_source_indices]
        self.fingertip_points_world_end   = self.fingertip_transformed_mocap.points_world_end[self.valid_source_indices]
        self.fingertip_normal_world       = self.fingertip_transformed_mocap.normal_world[self.valid_source_indices]


    def update_mocap(self):
        # ---- object ----
        self.__update_object_valid_mocap()
        self.__update_object_invalid_mocap()
        # ---- fingertip ----
        self.__update_fingertip_valid_mocap()
        self.__update_fingertip_invalid_mocap()

    def __update_object_valid_mocap(self):
        for i in range(len(self.valid_target_indices)):
            # print(f"valid idx = {i}")
            self.mocap_manager.set_pos(mocap_body_id = self.mocap_manager.name2id(f"{self.object_mocap_name}_start{i}"),  mocap_pos=self.object_points_world_start[i])
            self.mocap_manager.set_pos(mocap_body_id = self.mocap_manager.name2id(f"{self.object_mocap_name}_end{i}"),    mocap_pos=self.object_points_world_end[i])

    def __update_object_invalid_mocap(self):
        # print("==================================-")
        index_bias = len(self.valid_target_indices)
        for i in range(self.max_mocap_num - len(self.valid_target_indices)):
            i = i + index_bias
            # print(f"invalid idx = {i}")
            # import ipdb ; ipdb.set_trace()
            self.mocap_manager.set_pos(mocap_body_id = self.mocap_manager.name2id(f"{self.object_mocap_name}_start{i}"),  mocap_pos=np.zeros(3))
            self.mocap_manager.set_pos(mocap_body_id = self.mocap_manager.name2id(f"{self.object_mocap_name}_end{i}"),    mocap_pos=np.zeros(3))


    def __update_fingertip_valid_mocap(self):
        for i in range(len(self.valid_source_indices)):
            # print(f"valid idx = {i}")
            self.mocap_manager.set_pos(mocap_body_id = self.mocap_manager.name2id(f"{self.fingertip_mocap_name}_start{i}"),  mocap_pos=self.fingertip_points_world_start[i])
            self.mocap_manager.set_pos(mocap_body_id = self.mocap_manager.name2id(f"{self.fingertip_mocap_name}_end{i}"),    mocap_pos=self.fingertip_points_world_end[i])


    def __update_fingertip_invalid_mocap(self):
        # print("==================================-")
        index_bias = len(self.valid_source_indices)
        for i in range(self.max_mocap_num - len(self.valid_source_indices)):
            i = i + index_bias
            # print(f"invalid idx = {i}")
            # import ipdb ; ipdb.set_trace()
            self.mocap_manager.set_pos(mocap_body_id = self.mocap_manager.name2id(f"{self.fingertip_mocap_name}_start{i}"),  mocap_pos=np.zeros(3))
            self.mocap_manager.set_pos(mocap_body_id = self.mocap_manager.name2id(f"{self.fingertip_mocap_name}_end{i}"),    mocap_pos=np.zeros(3))


    # =============== getter ===============
    def get_object_points_world(self):
        return self.object_points_world_start

    def get_object_normals_world(self):
        return self.object_normal_world

    def get_fingertip_points_world(self):
        return self.fingertip_points_world_start

    def get_fingertip_normals_world(self):
        return self.fingertip_normal_world