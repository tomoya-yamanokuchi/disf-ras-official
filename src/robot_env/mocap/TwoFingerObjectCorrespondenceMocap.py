from scipy.spatial import KDTree
from .BoxObjectMocap import BoxObjectMocap
from .SingleObjectCorrespondenceMocap import SingleObjectCorrespondenceMocap
from service import correspondence_filtering_between_fingers
from value_object import TwoFingerPointAndNormal, SingleFingerPointAndNormal

class TwoFingerObjectCorrespondenceMocap:
    def __init__(self,
            object_mocap                      : BoxObjectMocap,
            right_object_correspondence_mocap : SingleObjectCorrespondenceMocap,
            left_object_correspondence_mocap  : SingleObjectCorrespondenceMocap,
        ):
        self.object_mocap = object_mocap
        self.right = right_object_correspondence_mocap
        self.left  = left_object_correspondence_mocap


    def update_correspondece(self):
        tree = KDTree(self.object_mocap.get_points_world())
        # ---
        self.right.update_correspondence_with_single_finger_filtering(tree)
        self.left.update_correspondence_with_single_finger_filtering(tree)
        right_nonoverlap, left_nonoverlap = correspondence_filtering_between_fingers(
            right_correspondence = self.right.correspondence,
            left_correspondence  = self.left.correspondence,
        )
        # ---
        self.right.update_correspondence_with_two_finger_filtering(right_nonoverlap)
        self.left.update_correspondence_with_two_finger_filtering(left_nonoverlap)

        # self.right.debug_update_correspondence_with_two_finger_filtering() # debug !!!!!!

        # ---
        self.right.update_mocap()
        self.left.update_mocap()

    def update_valid_points_and_normals(self):
        self.right.update_valid_points_and_normals()
        self.left.update_valid_points_and_normals()
        # ---
        self.right.update_mocap()
        self.left.update_mocap()

    def get_params_for_palm_optimization(self):
        right = SingleFingerPointAndNormal(
            pj  = self.right.get_fingertip_points_world(),
            qj  = self.right.get_object_points_world(),
            npj = self.right.get_fingertip_normals_world(),
            nqj = self.right.get_object_normals_world(),
        )
        left = SingleFingerPointAndNormal(
            pj  = self.left.get_fingertip_points_world(),
            qj  = self.left.get_object_points_world(),
            npj = self.left.get_fingertip_normals_world(),
            nqj = self.left.get_object_normals_world(),
        )
        return TwoFingerPointAndNormal(right=right, left=left)


    def get_object_params_for_palm_optimization(self):
        right = SingleFingerPointAndNormal(
            qj  = self.right.get_object_points_world(),
            nqj = self.right.get_object_normals_world(),
        )
        left = SingleFingerPointAndNormal(
            qj  = self.left.get_object_points_world(),
            nqj = self.left.get_object_normals_world(),
        )
        return TwoFingerPointAndNormal(right=right, left=left)


    def get_params_for_finger_optimization(self):
        return {
            # ---
            "p1_array"  : self.right.get_fingertip_points_world(),
            "q1_array"  : self.right.get_object_points_world(),
            "nq1_array" : self.right.get_object_normals_world(),
            # ---
            "p2_array"  : self.left.get_fingertip_points_world(),
            "q2_array"  : self.left.get_object_points_world(),
            "nq2_array" : self.left.get_object_normals_world(),
        }

