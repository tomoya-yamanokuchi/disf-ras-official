from .TwoFingertipBaseMocap import TwoFingertipBaseMocap
from .TwoFingertipTransformedMocap import TwoFingertipTransformedMocap
from .TwoFingerObjectCorrespondenceMocap import TwoFingerObjectCorrespondenceMocap
from .BoxObjectMocap import BoxObjectMocap


class MocapSet:
    def __init__(self,
            fingertip_base_mocap        : TwoFingertipBaseMocap,
            fingertip_transformed_mocap : TwoFingertipTransformedMocap,
            object_mocap                : BoxObjectMocap,
            object_correspondence_mocap : TwoFingerObjectCorrespondenceMocap,
        ):
        self.fingertip_base_mocap        = fingertip_base_mocap
        self.fingertip_transformed_mocap = fingertip_transformed_mocap
        self.object_mocap                = object_mocap
        self.object_correspondence_mocap = object_correspondence_mocap

