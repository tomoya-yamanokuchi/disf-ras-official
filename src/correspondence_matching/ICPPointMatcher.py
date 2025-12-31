import numpy as np
from typing import Tuple
from scipy.spatial import KDTree
from value_object import PointNormalUnitPairs
from value_object import PointNormalIndexUnitPairs
from domain_object.builder import DomainObject
from value_object import ICPFiltering
from scipy.spatial import KDTree


class ICPPointMatcher:
    def __init__(self, domain_object: DomainObject):
        pass

    def find_correspondences(self,
            source      : PointNormalIndexUnitPairs,
            target      : PointNormalUnitPairs,
            target_tree : KDTree,
        ) -> Tuple[PointNormalUnitPairs, np.ndarray]:
        # ---
        distances, indices           = target_tree.query(source.points)
        target_corresponding_points  = target.points[indices]
        target_corresponding_normals = target.normals[indices]

        # ----
        correspondences = PointNormalUnitPairs(
            points  = target_corresponding_points,
            normals = target_corresponding_normals,
        )
        return correspondences, distances, indices


