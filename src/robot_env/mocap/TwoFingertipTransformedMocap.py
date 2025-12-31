from .SingleFingertipTransformedMocap import SingleFingertipTransformedMocap
from value_object import GripperTransformationParams, SingleFingerPointAndNormal, TwoFingerPointAndNormal


class TwoFingertipTransformedMocap:
    def __init__(self,
            right_fingertip_transformed_mocap : SingleFingertipTransformedMocap,
            left_fingertip_transformed_mocap  : SingleFingertipTransformedMocap,
        ):
        self.right = right_fingertip_transformed_mocap
        self.left  = left_fingertip_transformed_mocap

    def update_transform_into_base(self, gripper_transform_params: GripperTransformationParams):
        self.right.update_transform_into_base(gripper_transform_params)
        self.left.update_transform_into_base(gripper_transform_params)

    def update_transform_into_self(self, gripper_transform_params: GripperTransformationParams):
        self.right.update_transform_into_self(gripper_transform_params)
        self.left.update_transform_into_self(gripper_transform_params)

    def get_right_points(self):
        return self.right.get_points_world()

    def get_left_points(self):
        return self.left.get_points_world()

    def get_params_for_palm_optimization(self):
        right = SingleFingerPointAndNormal(
            pj  = self.right.get_points_world(),
            npj = self.right.get_points_world(),
        )
        left = SingleFingerPointAndNormal(
            pj  = self.left.get_points_world(),
            npj = self.left.get_points_world(),
        )
        return TwoFingerPointAndNormal(right=right, left=left)

