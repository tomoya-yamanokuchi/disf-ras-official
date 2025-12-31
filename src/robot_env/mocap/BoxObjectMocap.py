import numpy as np
from service import transform
from service import generate_box_surface_points
from service import dictionalize_box_surface
from service import create_normals_object
from ..utils.GeomManager import GeomManager
from .MocapManager import MocapManager
from .BoxObjectMocapInstanceDict import BoxObjectMocapInstanceDict


class BoxObjectMocap:
    def __init__(self, paramsDict: BoxObjectMocapInstanceDict):
        self.geom           : GeomManager = paramsDict['geom']
        self.mocap_manager : MocapManager = paramsDict['mocap_manager']
        self.mocap_name_point_start       = paramsDict['object_name'] + "_mocap_start"
        self.mocap_name_point_end         = paramsDict['object_name'] + "_mocap_end"
        self.object_name                  = paramsDict['object_name']
        self.scale_param                  = paramsDict['scale_param']
        # ---
        self.id           = self.geom.name2id(paramsDict['object_name'])
        box_size          = self.geom.size(geom_id=self.id)
        self.points_local = generate_box_surface_points(box_size, paramsDict['resolution'])
        self.normal_local = create_normals_object(resolution=paramsDict['resolution'])
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
        self.rotation_matrix    = self.geom.xmat(geom_id=self.id) # (3, 3)
        self.translation_vector = self.geom.xpos(geom_id=self.id) # (3,)

    def __update_position_and_normal(self):
        # --- start points ---
        self.points_world_start      = transform(self.points_local, self.rotation_matrix, self.translation_vector)
        self.dict_points_world_start = dictionalize_box_surface(self.points_world_start)
        # --- normals ---
        self.normal_world            = np.dot(self.normal_local, self.rotation_matrix.T)
        # --- end points ---
        self.points_world_end        = self.points_world_start + (self.normal_world * self.scale_param)
        self.dict_points_world_end   = dictionalize_box_surface(self.points_world_end)

    def __update_mocap(self):
        for i in range(self.dict_points_world_start["num_point"]):
            # --- start ----
            self.mocap_manager.set_pos(mocap_body_id=self.mocap_manager.name2id( f"box_z_plus_mocap_start{i}"),  mocap_pos = self.dict_points_world_start["z_plus"][i])
            self.mocap_manager.set_pos(mocap_body_id=self.mocap_manager.name2id(f"box_z_minus_mocap_start{i}"),  mocap_pos = self.dict_points_world_start["z_minus"][i])
            self.mocap_manager.set_pos(mocap_body_id=self.mocap_manager.name2id( f"box_x_plus_mocap_start{i}"),  mocap_pos = self.dict_points_world_start["x_plus"][i])
            self.mocap_manager.set_pos(mocap_body_id=self.mocap_manager.name2id(f"box_x_minus_mocap_start{i}"),  mocap_pos = self.dict_points_world_start["x_minus"][i])
            self.mocap_manager.set_pos(mocap_body_id=self.mocap_manager.name2id( f"box_y_plus_mocap_start{i}"),  mocap_pos = self.dict_points_world_start["y_plus"][i])
            self.mocap_manager.set_pos(mocap_body_id=self.mocap_manager.name2id(f"box_y_minus_mocap_start{i}"),  mocap_pos = self.dict_points_world_start["y_minus"][i])
            # --- end ----
            self.mocap_manager.set_pos(mocap_body_id=self.mocap_manager.name2id( f"box_z_plus_mocap_end{i}"),    mocap_pos = self.dict_points_world_end["z_plus"] [i])
            self.mocap_manager.set_pos(mocap_body_id=self.mocap_manager.name2id(f"box_z_minus_mocap_end{i}"),    mocap_pos = self.dict_points_world_end["z_minus"][i])
            self.mocap_manager.set_pos(mocap_body_id=self.mocap_manager.name2id( f"box_x_plus_mocap_end{i}"),    mocap_pos = self.dict_points_world_end["x_plus"] [i])
            self.mocap_manager.set_pos(mocap_body_id=self.mocap_manager.name2id(f"box_x_minus_mocap_end{i}"),    mocap_pos = self.dict_points_world_end["x_minus"][i])
            self.mocap_manager.set_pos(mocap_body_id=self.mocap_manager.name2id( f"box_y_plus_mocap_end{i}"),    mocap_pos = self.dict_points_world_end["y_plus"] [i])
            self.mocap_manager.set_pos(mocap_body_id=self.mocap_manager.name2id(f"box_y_minus_mocap_end{i}"),    mocap_pos = self.dict_points_world_end["y_minus"][i])

    def get_points_world(self):
        return self.points_world_start

