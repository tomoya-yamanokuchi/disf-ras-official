from domain_object.builder import DomainObject
from value_object import IPFOErrors
import numpy as np
from print_color import print

class ISF_LoopCriteria:
    def __init__(self, domain_object: DomainObject):
        self.delta_e        = domain_object.delta_e
        self.text_logger    = domain_object.text_logger
        self.max_ipfo_count = domain_object.config_isf.max_ipfo_count
        # ---
        self.count = None

    def reset_count(self):
        self.count = 0


    def add_count(self):
        self.count += 1

    def evaluate(self,
            e_p: IPFOErrors,
            e_t: IPFOErrors
        ):
        if self.count >= self.max_ipfo_count:
            print(f"iteration is reached to maximum count! (={self.count})",
                tag="IPFO eval", color="yellow", tag_color="yellow")
            return False

        delta_e = np.abs(e_p.total - e_t.total)
        # ---
        self.text_logger.delta_e(delta_e)
        # ---
        return delta_e > self.delta_e

