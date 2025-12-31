import numpy as np
from value_object import PointNormalUnitPairs
from domain_object.builder import SelfContainedDomainObjectBuilder as DomainObject


class ApproachDirectionLeastSquareParameter:
    def __init__(self, domain_object: DomainObject):
        self.n_app = domain_object.n_app
        self.beta  = domain_object.beta

    def compute(self, n_z: np.ndarray):
        # ----------
        Aa =   self.beta * np.cross(n_z, self.n_app)
        ba = - self.beta * (np.dot(n_z, self.n_app) - 1)
        # ----------
        Aa = np.array(Aa).reshape(1, -1)
        ba = np.array([ba]).reshape(1,)
        # ----------
        # import ipdb ; ipdb.set_trace()
        return Aa, ba


