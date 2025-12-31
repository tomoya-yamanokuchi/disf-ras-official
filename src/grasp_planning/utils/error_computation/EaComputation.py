import numpy as np
import os
from value_object import IPFOErrors
from domain_object.builder import DomainObject
from ...error.approach_alignment_error import approach_alignment_error
from .ErrorHistory import ErrorHistory


class EaComputation:
    def __init__(self, domain_object: DomainObject):
        self.name = "Ea"
        # ---
        self.beta             = domain_object.beta
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

    def save_history(self):
        np.save(
            file = os.path.join(self.results_save_dir, f"{self.name}.npy"),
            arr  = self.get_history(),
        )

    def compute(self,
            n_z  : np.ndarray,
            history_append: bool = False,
        ) -> IPFOErrors:
        # -------
        Ea = np.sum(approach_alignment_error(n_z=n_z, n_app=self.n_app)**2)
        # -------
        self.text_logger.error_Ea(Ea)
        # -------
        if history_append:
            self.history.append(Ea)
        # -------
        return Ea

    def compute_with_weight(self,
            n_z  : np.ndarray,
            history_append : bool = False,
        ) -> IPFOErrors:
        return (self.beta**2) * self.compute(n_z, history_append)

