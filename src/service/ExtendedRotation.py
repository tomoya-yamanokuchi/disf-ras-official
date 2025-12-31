from __future__ import annotations
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation
from .small_angle_rotation_matix import small_angle_rotation_matix


class ExtendedRotation(Rotation):
    def as_skew(self) -> np.ndarray:
        rotvec = self.as_rotvec()
        assert rotvec.shape == (3,) # not compatible with batch data
        return small_angle_rotation_matix(r=rotvec)

    def as_rodrigues(self) -> np.ndarray:
        return self.as_matrix()

    def as_quat_scalar_first(self) -> np.ndarray:
        quat              = self.as_quat() # scalar last
        quat_scalar_first = np.hstack([quat[-1:], quat[:-1]])
        return quat_scalar_first

    @classmethod
    def from_euler(cls, euler: npt.ArrayLike) -> ExtendedRotation:
        # ---

        instance = super().from_euler(seq='xyz', angles=euler)
        # ---
        return cls(instance.as_quat())

    @classmethod
    def from_rotvec(cls, rotvec: npt.ArrayLike) -> ExtendedRotation:
        # ---
        instance = super().from_rotvec(rotvec)
        # ---
        return cls(instance.as_quat())

    @classmethod
    def from_quat(cls, quat: npt.ArrayLike) -> ExtendedRotation:
        """
            quat : scalar-first order - (w, x, y, z)
        """
        quat_scalar_last = np.hstack([quat[1:], quat[:1]])
        instance = super().from_quat(quat_scalar_last)
        # ---
        return cls(instance.as_quat())

    @classmethod
    def from_matrix(cls, matrix: npt.ArrayLike) -> ExtendedRotation:
        # ---
        instance = super().from_matrix(matrix)
        # ---
        return cls(instance.as_quat())
