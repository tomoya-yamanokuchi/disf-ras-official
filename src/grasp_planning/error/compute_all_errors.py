import numpy as np
from .point2plane_error import point2plane_error
from .point2point_error import point2point_error
from .normal_alignment_error import normal_alignment_error
from .approach_alignment_error import approach_alignment_error
from service import format_vector
from value_object import IPFOErrors
from value_object import PointNormalUnitPairs


def compute_all_errors(
        aligned_source: PointNormalUnitPairs,
        target        : PointNormalUnitPairs,
        alpha         : float,
        verbose       : bool = False,
    ):
    E_point2point = np.sum(point2point_error(aligned_source, target))  # not used in optimization
    # ---
    Ep            = np.sum(point2plane_error(aligned_source, target)**2)
    En            = np.sum(normal_alignment_error(aligned_source, target)**2)
    # --------------------------------
    Err = Ep + (alpha**2 * En)
    # import ipdb ; ipdb.set_trace()
    # --------------------------------
    if verbose:
        print(f"Err={format_vector([Err], decimal_places=4)} | E_point2point={format_vector([E_point2point], decimal_places=4)} | Ep={format_vector([Ep], decimal_places=4)} | En={format_vector([En], decimal_places=4)}")
        # import ipdb ; ipdb.set_trace()
    # -------------------
    return IPFOErrors(
        total            = Err,
        point2point      = E_point2point,
        point2plaine     = Ep,
        normal_alignment = En,
    )
