from typing import TypedDict
from .MocapManager import MocapManager
from .SingleFingertipTransformedMocap import SingleFingertipTransformedMocap
from .BoxObjectMocap import BoxObjectMocap


class SingleObjectCorrespondenceMocapInstanceDict(TypedDict):
    mocap_manager                : MocapManager
    fingertip_transformed_mocap  : SingleFingertipTransformedMocap
    object_mocap                 : BoxObjectMocap
    threshold                    : float
    object_mocap_name            : str
    fingertip_mocap_name         : str
