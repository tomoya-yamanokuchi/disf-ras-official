import numpy as np
from service import transform_gripper
from domain_object.builder import DomainObject
from value_object import PointNormalUnitPairs


class FingertipTransformation:
    def __init__(self, domain_object: DomainObject):
        self._finger_indices = domain_object.finger_indices
        self.v0 = np.array(domain_object.v0)
        # ---
        self.R       = None
        self.t       = None
        self.delta_d = None
        self.Rg      = None
        self.j       = 1

    def update_R(self, R: np.ndarray):
        self.R = R

    def update_t(self, t: np.ndarray):
        self.t = t

    def update_Rg(self, Rg: np.ndarray):
        self.Rg = Rg

    def update_delta_d(self, delta_d: np.ndarray):
        self.delta_d = delta_d

    def __transform_points(self, source_points: np.ndarray):
        num_correspondences = source_points.shape[0]
        finger_indices      = np.zeros(num_correspondences) + self.j
        # -------
        return transform_gripper(
            source_points     = source_points,
            # finger_indices    = self._finger_indices,
            finger_indices    = finger_indices,
            gripper_normal    = self.Rg @ self.g,
            # -----
            rotation_matrix   = self.R,
            translation       = self.t,
            delta_d           = self.delta_d,
        )

    def __transform_normals(self, source_normals):
        return np.dot(source_normals, self.R.T)

    def transform(self, source: PointNormalUnitPairs):
        return PointNormalUnitPairs(
            points  = self.__transform_points(source.points),
            normals = self.__transform_normals(source.normals)
        )
