import numpy as np
from typing import NamedTuple
from scipy.spatial import KDTree
from value_object import TargetPointNormalIndexPairs
from value_object import PointNormalUnitPairs, PointNormalIndexUnitPairs
from domain_object.builder import DomainObject
from value_object import ICPResult
from .ICPPointMatcher import ICPPointMatcher
from value_object import ICPFiltering
from .filter_duplicates import filter_duplicates
from .filter_by_angle_threshold import filter_by_angle_threshold
from .filter_by_finger_normals import filter_by_finger_normals
from .orient_target_normals_towards_source import orient_target_normals_towards_source

class ICPPointMatcherWithNormals:
    def __init__(self, domain_object: DomainObject):
        self.matcher         = ICPPointMatcher(domain_object)
        self.angle_threshold = np.deg2rad(domain_object.config_icp.angle_threshold_degree)
        # import ipdb ; ipdb.set_trace()

    def set_target(self, target : PointNormalUnitPairs):
        """
        This is the whole taregt object point cloud data
        """
        self.target      = target
        self.target_tree = KDTree(self.target.points)


    def find_correspondences(self, source: PointNormalIndexUnitPairs) -> ICPResult:
        target_correspondences, distances, indices = self.matcher.find_correspondences(
            source, self.target, self.target_tree
        )
        # -----
        filtered_result = filter_duplicates(
            source                         = source,
            target_correspondences         = target_correspondences,
            target_correspondences_indices = indices,
            distances                      = distances,
        )
        # -----
        filtered_result = filter_by_finger_normals(
            filtered_result = filtered_result,
            angle_threshold = self.angle_threshold,
        )
        # # -----
        filtered_result = filter_by_angle_threshold(
            filtered_result = filtered_result,
            angle_threshold = self.angle_threshold
        )
        # ------
        print(f"num_target = {filtered_result.filtered_target.indices.shape[0]}")
        # import ipdb; ipdb.set_trace()
        # ------
        return ICPResult(
            source = filtered_result.filtered_source,
            target = filtered_result.filtered_target,
            num_correspondences = filtered_result.filtered_target.indices.shape[0],
        )
