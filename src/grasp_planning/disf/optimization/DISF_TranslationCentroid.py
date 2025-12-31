import numpy as np
from service import calculate_centroid
from value_object import SourcePointSurfaceSet, TargetPointSurfaceSet
from domain_object.builder import DomainObject
from service import transform_source_set


class DISF_TranslationCentroid:
    def __init__(self, domain_object: DomainObject):
        self.disf_palm_R_opt      = domain_object.disf_palm_R_opt
        # ----
        self.isf_visualizer       = domain_object.isf_visualizer
        self.text_logger          = domain_object.text_logger
        self.error                = domain_object.error
        self.verbose              = domain_object.verbose
        self.finger_indices       = domain_object.finger_indices
        self.v0                   = domain_object.v0
        self.n_z0                 = domain_object.n_z0
        # ---
        self.call_level = 0


    def optimize(self,
            source_set : SourcePointSurfaceSet,
            target_set : TargetPointSurfaceSet,
            # ----
            n_z        : np.ndarray,
            v          : np.ndarray,
        ):
        # -------------------------------------
        delta_d0    = 0.0
        R_I         = np.eye(3)
        # ------------- refinement -------------
        """ old (all surface)
            source_center      = calculate_centroid(source_set.surface.points, keepdims=False)
            target_center      = calculate_centroid(target_set.contact_surface.points, keepdims=False)
        """

        """ """  """ """ # new (contact surface)
        source_center = calculate_centroid(source_set.correspondence.points, keepdims=False)
        target_center = calculate_centroid(target_set.correspondence.points, keepdims=False)
        """ """
        # import ipdb; ipdb.set_trace()

        t_shift_centroid   = (target_center - source_center)
        # -------- compute transformation -------
        aligned_source_set = transform_source_set(source_set, R_I, t_shift_centroid, delta_d0, v)
        # ----------- error computation -----------
        errors             = self.error.compute(aligned_source_set, target_set, n_z)
        # ---------------- textlog -----------------
        self.text_logger.palm_trans_centroid_finished(t_shift_centroid)
        # ---------------- visualization -----------
        self.isf_visualizer.visualize(
            source_set = aligned_source_set,
            target_set = target_set,
            n_z        = n_z,
            call_level = self.call_level,
            title="[DIPFO] t*: trans centroid opt",
        )
        # ------------------------------------------
        return aligned_source_set, t_shift_centroid

