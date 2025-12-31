import os
import numpy as np
from value_object import IPFOErrors
from domain_object.builder import DomainObject
from value_object import SourcePointSurfaceSet
from value_object import TargetPointSurfaceSet
# -----
from ...error.point2plane_error import point2plane_error
from .ErrorHistory import ErrorHistory


class EpComputation:
    def __init__(self, domain_object: DomainObject):
        self.name = "Ep"
        # ---
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
        # import ipdb; ipdb.set_trace()

    def compute(self,
            source_set    : SourcePointSurfaceSet,
            target_set    : TargetPointSurfaceSet,
            history_append: bool = False,
        ) -> IPFOErrors:
        # -------
        source = source_set.correspondence
        target = target_set.correspondence
        # -------
        Ep = np.sum(point2plane_error(source, target)**2)
        # -------
        self.text_logger.error_Ep(Ep=Ep)
        # -------
        if history_append:
            self.history.append(Ep)
        # -------
        return Ep

