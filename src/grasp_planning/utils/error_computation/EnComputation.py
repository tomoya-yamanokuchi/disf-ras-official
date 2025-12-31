import os
import numpy as np
from value_object import IPFOErrors
from domain_object.builder import DomainObject
from value_object import SourcePointSurfaceSet
from value_object import TargetPointSurfaceSet
# -----
from ...error.normal_alignment_error import normal_alignment_error
from .ErrorHistory import ErrorHistory


class EnComputation:
    def __init__(self, domain_object: DomainObject):
        self.name = "En"
        # ---
        self.alpha            = domain_object.alpha
        self.error_verbose    = domain_object.verbose.textlog.error
        self.text_logger      = domain_object.text_logger
        self.results_save_dir = domain_object.results_save_dir
        self.method_name      = domain_object.method_name
        self.model_name       = domain_object.model_name
        self.n_app            = domain_object.n_app
        # ---
        self.history          = ErrorHistory()

    def get_history(self):
        return np.hstack(self.history.hist)

    def save_history(self, tag: str = ""):
        name = self.name + "_" + tag
        np.save(
            file = os.path.join(self.results_save_dir, f"{name}.npy"),
            arr  = self.get_history(),
        )

    def compute(self,
            source_set: SourcePointSurfaceSet,
            target_set: TargetPointSurfaceSet,
            history_append: bool = False,
        ) -> IPFOErrors:
        # -------
        source = source_set.correspondence
        target = target_set.correspondence
        # -------
        En = np.sum(normal_alignment_error(source, target)**2)
        # -------
        self.text_logger.error_En(En=En)
        # -------
        if history_append:
            self.history.append(En)
        # -------
        return En


    def compute_with_weight(self,
            source_set: SourcePointSurfaceSet,
            target_set: TargetPointSurfaceSet,
            history_append : bool = False,
        ) -> IPFOErrors:
        return (self.alpha**2) * self.compute(source_set, target_set, history_append)
