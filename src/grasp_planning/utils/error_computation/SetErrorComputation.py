import os
import numpy as np
from value_object import IPFOErrors
from domain_object.builder import DomainObject
from value_object import SourcePointSurfaceSet
from value_object import TargetPointSurfaceSet
from .ErrorHistory import ErrorHistory


class SetErrorComputation:
    def __init__(self, domain_object: DomainObject):
        self.name = "E_total"
        # ---
        self.Ep = domain_object.Ep
        self.En = domain_object.En
        self.Ea = domain_object.Ea
        # ---
        self.results_save_dir = domain_object.results_save_dir
        self.method_name      = domain_object.method_name
        self.model_name       = domain_object.model_name
        self.n_app            = domain_object.n_app
        # ---
        self.config_pc_data = domain_object.config_pc_data
        # ---
        self.history          = ErrorHistory()

    def get_history(self):
        return np.hstack(self.history.hist)

    def save_history(self, sub=False):
        # --------------------
        beta     = self.En.alpha
        tag_beta = f"beta={beta:.0e}"
        name     = self.name + "_" + tag_beta
        # if "box" in self.model_name:
        #     tag_size = f"object_half_size={self.config_pc_data.object_half_size:.0e}"
        #     name     = name + "_" + tag_size
        # --------------------
        np.save(
            file = os.path.join(self.results_save_dir, f"{name}.npy"),
            arr  = self.get_history(),
        )
        # ---------------------
        if sub:
            tag = tag_beta + "_" + tag_size
            self.Ep.save_history(tag=tag)
            self.En.save_history(tag=tag)

    def compute(self,
            source_set        : SourcePointSurfaceSet,
            target_set        : TargetPointSurfaceSet,
            n_z               : np.ndarray,
            history_append    : bool = False,
            history_append_sub: bool = False,
        ) -> IPFOErrors:
        # -------
        Ep       = self.Ep.compute(source_set, target_set, history_append_sub)
        beta_En  = self.En.compute_with_weight(source_set, target_set, history_append_sub)
        alpha_Ea = self.Ea.compute_with_weight(n_z, history_append_sub)
        # -------
        E_total  = (Ep + beta_En + alpha_Ea)
        # -------
        if history_append:
            self.history.append(E_total)
        # -------
        return IPFOErrors(
            total              = E_total,
            point2plaine       = Ep,
            normal_alignment   = beta_En,
            approach_alignment = alpha_Ea,
        )
