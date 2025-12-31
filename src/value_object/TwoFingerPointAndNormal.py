from typing import TypedDict
from .SingleFingerPointAndNormal import SingleFingerPointAndNormal


class TwoFingerPointAndNormal(TypedDict):
    right : SingleFingerPointAndNormal
    left  : SingleFingerPointAndNormal