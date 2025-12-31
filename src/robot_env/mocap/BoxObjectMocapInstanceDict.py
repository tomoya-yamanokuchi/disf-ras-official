from typing import TypedDict
import numpy as np
from ..utils.GeomManager import GeomManager
from .MocapManager import MocapManager


class BoxObjectMocapInstanceDict(TypedDict):
    geom           : GeomManager
    mocap_manager  : MocapManager
    object_name    : str
    scale_param    : float
    resolution     : int
