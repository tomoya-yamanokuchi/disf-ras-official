from typing import TypedDict
import numpy as np
from ..utils.GeomManager import GeomManager
from ..utils.BodyManager import BodyManager
from .MocapManager import MocapManager
import mujoco


class SingleFingertipBaseMocapInstanceDict(TypedDict):
    data               : mujoco.MjData
    geom               : GeomManager
    body               : BodyManager
    mocap_manager      : MocapManager
    mocap_name         : str
    normal_gripper     : np.ndarray
    scale_param        : float
    name_body_hand     : str
    name_geom_fingertip: str
