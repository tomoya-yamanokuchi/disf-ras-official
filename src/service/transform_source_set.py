import numpy as np
from .transform_fingertip import transform_fingertip
from value_object.SourcePointSurfaceSet import SourcePointSurfaceSet


def transform_source_set(
        source_set : SourcePointSurfaceSet,
        # ---
        R      : np.ndarray, # (3, 3)
        t      : np.ndarray, # (3,)
        delta_d: float,
        # ---
        v      : np.ndarray, # (3,)
    ):
    # -------
    return SourcePointSurfaceSet(
        correspondence = transform_fingertip(source_set.correspondence, R, t, delta_d, v),
        surface        = transform_fingertip(       source_set.surface, R, t, delta_d, v),
    )
