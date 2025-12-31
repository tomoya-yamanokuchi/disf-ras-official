import numpy as np
from .point2plane_error import point2plane_error
from .normal_alignment_error import normal_alignment_error
from .approach_alignment_error import approach_alignment_error
from service import format_vector
from value_object import AppJointIPFOErrors
from value_object import PointNormalUnitPairs

def compute_all_errors_app_joint(
        aligned_source: PointNormalUnitPairs,
        target        : PointNormalUnitPairs,
        n_z           : np.ndarray,
        n_app         : np.ndarray,
        alpha         : float,
        beta          : float,
        verbose       : bool = False,
    ):
    Ep  = np.sum(point2plane_error(aligned_source, target)**2)
    En  = np.sum(normal_alignment_error(aligned_source, target)**2)
    Ea  = (approach_alignment_error(n_z, n_app)**2)
    # --------------------------------
    Err = Ep + (alpha**2 * En) + (beta**2 * Ea)
    # import ipdb ; ipdb.set_trace()
    # --------------------------------
    # import ipdb ; ipdb.set_trace()
    if verbose:
        print(f"Err={format_vector([Err], decimal_places=4)} | Ep={format_vector([Ep], decimal_places=4)} | En={format_vector([En], decimal_places=4)} | Ea={format_vector([Ea], decimal_places=4)}")
    # -------------------
    return AppJointIPFOErrors(
        total              = Err,
        point2plaine       = Ep,
        normal_alignment   = En,
        approach_alignment = Ea,
    )
