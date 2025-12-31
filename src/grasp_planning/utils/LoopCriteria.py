from domain_object.builder import DomainObject
import numpy as np

class LoopCriteria:
    def __init__(self, domain_object: DomainObject):
        self.epsilon     = domain_object.epsilon
        self.max_iter    = domain_object.config_isf.max_iter
        self.text_logger = domain_object.text_logger

    def evaluate(self, eta: float, es_prev: float, iter_count: int):
        # イテレーション数が最大値を超えた場合は False を返す
        if iter_count >= self.max_iter:
            self.text_logger.maximum_isf_iteration(iter_count, self.max_iter)
            # import ipdb ; ipdb.set_trace()
            return True

        if np.isnan(eta) and (es_prev == 0.0):
            """
            This prevent the bad initial ratation cases where
                (1) n_z and n_app are parallel     (0 [deg] difference)
                (2) n_z and n_app are antiparallel (eg. 180 or -180 or ... [deg]) difference)
            """
            return True
        # -----------
        # print("eta = ", eta)
        lower = (1 - self.epsilon)
        upper = (1 + self.epsilon)
        return (lower <= eta <= upper)

