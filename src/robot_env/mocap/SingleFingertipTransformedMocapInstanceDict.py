from typing import TypedDict
from .MocapManager import MocapManager
from .SingleFingertipBaseMocap import SingleFingertipBaseMocap


class SingleFingertipTransformedMocapInstanceDict(TypedDict):
    mocap_manager        : MocapManager
    mocap_name           : str
    fingertip_base_mocap : SingleFingertipBaseMocap
    scale_param          : float
