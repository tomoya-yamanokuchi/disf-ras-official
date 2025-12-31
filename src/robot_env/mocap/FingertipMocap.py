import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as R
from service import transform, generate_surface_points_panda_hand
from service import generate_surface_points_panda_hand
from service import generate_box_surface_points
from service import dictionalize_box_surface
from service import create_normals_object
from service import transform_gripper
from service import correspondence_filtering
from ..instance.hand.HandBaseEnv import DexterousHandEnv



class FingertipMocap:
    def __init__(self, geom):
        self.geom = geom
        # ---
        self.id_right_fingertip = self.geom.name2id("right_fingertip_center")
        self.id_left_fingertip  = self.geom.name2id("left_fingertip_center")
        self.set_local_points_and_normals()
        # -

    def set_local_points_and_normals(self):
        self.right_fingertip_points_local = generate_surface_points_panda_hand()
        self.left_fingertip_points_local  = generate_surface_points_panda_hand()
        # 顎の開閉方向を x 軸方向に仮定
        self.right_fingertip_normal_local = np.array([0, -1, 0])
        self.scale_param                  = 0.005

    def update_base_fingertip_mocap(self):
        self._update_fingertip_posture()
        self.__update_world_base_fingertip_position_and_normal()
        self.__update_world_base_fingertip_mocap()


    def _update_fingertip_posture(self):
        # @ right finger
        self.right_fingertip_rot   = self.geom.xmat(geom_id=self.id_right_fingertip)
        self.right_fingertip_trans = self.geom.xpos(geom_id=self.id_right_fingertip)
        # @ translation
        self.__left_fingertip_rot    = self.geom.xmat(geom_id=self.id_left_fingertip)
        self.__left_fingertip_trans  = self.geom.xpos(geom_id=self.id_left_fingertip)


    def __update_world_base_fingertip_position_and_normal(self):
        # --- start points ---
        self.__right_finger_points_world_start = transform(self.right_finger_points_local, self.__right_fingertip_rot, self.__right_fingertip_trans)
        self.__left_finger_points_world_start  = transform(self.left_finger_points_local,  self.__left_fingertip_rot,  self.__left_fingertip_trans)
        # --- normals ---
        self.__normal_right_finger_global      = np.dot(self.right_finger_normal_local, self.__right_fingertip_rot.T)
        self.__normal_left_finger_global       = np.dot(self.right_finger_normal_local, self.__left_fingertip_rot.T)
        # --- end points ---
        self.__right_finger_points_world_end   = (self.__right_finger_points_world_start + (self.__normal_right_finger_global * self.scale_param))
        self.__left_finger_points_world_end    = (self.__left_finger_points_world_start  + (self.__normal_left_finger_global  * self.scale_param))


    def __update_world_base_fingertip_mocap(self):
        for i in range(len(self.__right_finger_points_world_start)):
            # start
            self.mocap.set_pos(mocap_body_id=self.mocap.name2id(f"right_finger_mocap_base_start{i}"), mocap_pos=self.__right_finger_points_world_start[i])
            self.mocap.set_pos(mocap_body_id=self.mocap.name2id(f"left_finger_mocap_base_start{i}"),  mocap_pos=self.__left_finger_points_world_start[i])
            # end
            self.mocap.set_pos(mocap_body_id=self.mocap.name2id(f"right_finger_mocap_base_end{i}"),   mocap_pos=self.__right_finger_points_world_end[i])
            self.mocap.set_pos(mocap_body_id=self.mocap.name2id(f"left_finger_mocap_base_end{i}"),    mocap_pos=self.__left_finger_points_world_end[i])


    def __update_world_transformed_fingertip_position_and_normal(self,
            rotvec     : np.ndarray, # (3,)
            translation: np.ndarray, # (3,)
            delta_d    : np.ndarray, # (1,)
        ):
        rotation        = R.from_rotvec(rotvec)
        rotation_matrix = rotation.as_matrix()
        # --- start ---
        self.__right_fingertip_transformed_start = transform_gripper(self.__right_finger_points_world_start, rotation_matrix, translation, delta_d, j=1)
        self.__left_fingertip_transformed_start  = transform_gripper(self.__left_finger_points_world_start,  rotation_matrix, translation, delta_d, j=2)
        # --- end ---
        self.__right_fingertip_transformed_end   = transform_gripper(self.__right_finger_points_world_end,   rotation_matrix, translation, delta_d, j=1)
        self.__left_fingertip_transformed_end    = transform_gripper(self.__left_finger_points_world_end,    rotation_matrix, translation, delta_d, j=2)


    def __update_world_transformed_fingertip_mocap(self):
        for i in range(len(self.__right_fingertip_transformed_start)):
            # --- start ---
            self.mocap.set_pos(mocap_body_id = self.mocap.name2id(f"right_finger_mocap_transformed_start{i}"),  mocap_pos=self.__right_fingertip_transformed_start[i])
            self.mocap.set_pos(mocap_body_id = self.mocap.name2id( f"left_finger_mocap_transformed_start{i}"),  mocap_pos=self.__left_fingertip_transformed_start[i])
            # --- start ---
            self.mocap.set_pos(mocap_body_id = self.mocap.name2id(f"right_finger_mocap_transformed_end{i}"),    mocap_pos=self.__right_fingertip_transformed_end[i])
            self.mocap.set_pos(mocap_body_id = self.mocap.name2id( f"left_finger_mocap_transformed_end{i}"),    mocap_pos=self.__left_fingertip_transformed_end[i])


    def __update_object_correspondence(self):
        tree = KDTree(self.__object_points_world_start)
        # --- compute nearest neighbor with KDTree ---
        distances_right, indices_right = tree.query(self.__right_fingertip_transformed_start)
        distances_left,  indices_left  = tree.query(self.__left_fingertip_transformed_start)
        # --- filtering out with duplicate index ----
        self.__object_correspondence_right_start, _, _, valid_index_right = correspondence_filtering(self.__object_points_world_start, self.__right_fingertip_transformed_start, distances_right, indices_right)
        self.__object_correspondence_left_start , _, _, valid_index_left  = correspondence_filtering(self.__object_points_world_start, self.__left_fingertip_transformed_start , distances_left , indices_left )
        # --- end ---
        # import ipdb ; ipdb.set_trace()
        self.__object_correspondence_right_end = self.__object_points_world_end[valid_index_right]
        self.__object_correspondence_left_end  = self.__object_points_world_end[valid_index_left]
        # ---- normals ----
        self.__object_correspondence_right_fingertip_normal = self.__object_normals_world[valid_index_right]
        self.__object_correspondence_left_fingertip_normal  = self.__object_normals_world[valid_index_left]


    def __update_object_correspondence_mocap(self):
        for i in range(len(self.__object_correspondence_right_start)):
            # --- start ---
            self.mocap.set_pos(mocap_body_id = self.mocap.name2id(f"right_finger_correspondence_mocap_start{i}"),  mocap_pos=self.__object_correspondence_right_start[i])
            self.mocap.set_pos(mocap_body_id = self.mocap.name2id( f"left_finger_correspondence_mocap_start{i}"),  mocap_pos=self.__object_correspondence_left_start[i])
            # --- start ---
            self.mocap.set_pos(mocap_body_id = self.mocap.name2id(f"right_finger_correspondence_mocap_end{i}"),    mocap_pos=self.__object_correspondence_right_end[i])
            self.mocap.set_pos(mocap_body_id = self.mocap.name2id( f"left_finger_correspondence_mocap_end{i}"),    mocap_pos=self.__object_correspondence_left_end[i])
        # import ipdb ; ipdb.set_trace()

    def __update_object_posture(self):
        self.__object_rotation    = self.geom.xmat(self.id_object)
        self.__object_translation = self.geom.xpos(self.id_object)


    def __update_world_object_position_and_normal(self):
        # --- start points ---
        self.__object_points_world_start       = transform(self.object_points_local, self.__object_rotation, self.__object_translation)
        self.__dict_object_points_world_start  = dictionalize_box_surface(self.__object_points_world_start)
        # --- normals ---
        self.__object_normals_world            = np.dot(self.object_normals_local, self.__object_rotation.T)
        # --- end points ---
        self.__object_points_world_end         = self.__object_points_world_start + (self.__object_normals_world * self.scale_param)
        self.__dict_object_points_world_end    = dictionalize_box_surface(self.__object_points_world_end)


    def __update_world_object_mocap(self):
        for i in range(self.__dict_object_points_world_start["num_point"]):
            # --- start ----
            self.mocap.set_pos(mocap_body_id=self.mocap.name2id( f"box_z_plus_mocap_start{i}"),  mocap_pos = self.__dict_object_points_world_start["z_plus"][i])
            self.mocap.set_pos(mocap_body_id=self.mocap.name2id(f"box_z_minus_mocap_start{i}"),  mocap_pos = self.__dict_object_points_world_start["z_minus"][i])
            self.mocap.set_pos(mocap_body_id=self.mocap.name2id( f"box_x_plus_mocap_start{i}"),  mocap_pos = self.__dict_object_points_world_start["x_plus"][i])
            self.mocap.set_pos(mocap_body_id=self.mocap.name2id(f"box_x_minus_mocap_start{i}"),  mocap_pos = self.__dict_object_points_world_start["x_minus"][i])
            self.mocap.set_pos(mocap_body_id=self.mocap.name2id( f"box_y_plus_mocap_start{i}"),  mocap_pos = self.__dict_object_points_world_start["y_plus"][i])
            self.mocap.set_pos(mocap_body_id=self.mocap.name2id(f"box_y_minus_mocap_start{i}"),  mocap_pos = self.__dict_object_points_world_start["y_minus"][i])
            # --- end ----
            self.mocap.set_pos(mocap_body_id=self.mocap.name2id( f"box_z_plus_mocap_end{i}"),  mocap_pos = self.__dict_object_points_world_end["z_plus"] [i])
            self.mocap.set_pos(mocap_body_id=self.mocap.name2id(f"box_z_minus_mocap_end{i}"),  mocap_pos = self.__dict_object_points_world_end["z_minus"][i])
            self.mocap.set_pos(mocap_body_id=self.mocap.name2id( f"box_x_plus_mocap_end{i}"),  mocap_pos = self.__dict_object_points_world_end["x_plus"] [i])
            self.mocap.set_pos(mocap_body_id=self.mocap.name2id(f"box_x_minus_mocap_end{i}"),  mocap_pos = self.__dict_object_points_world_end["x_minus"][i])
            self.mocap.set_pos(mocap_body_id=self.mocap.name2id( f"box_y_plus_mocap_end{i}"),  mocap_pos = self.__dict_object_points_world_end["y_plus"] [i])
            self.mocap.set_pos(mocap_body_id=self.mocap.name2id(f"box_y_minus_mocap_end{i}"),  mocap_pos = self.__dict_object_points_world_end["y_minus"][i])


    def get_right_fingertip_transformed_points(self):
        return self.__right_fingertip_transformed_start

    def get_left_fingertip_transformed_points(self):
        return self.__left_fingertip_transformed_start

    def get_object_correspondence_right_fingertip_points(self):
        return self.__object_correspondence_right_start

    def get_object_correspondence_left_fingertip_points(self):
        return self.__object_correspondence_left_start

    def get_object_correspondence_right_fingertip_normal(self):
        return self.__object_correspondence_right_fingertip_normal

    def get_object_correspondence_left_fingertip_normal(self):
        return self.__object_correspondence_left_fingertip_normal
