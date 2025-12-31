from .instance.arm.ManipulatorGraspEnv import ManipulatorGraspEnv
from .instance.hand.HandDemoEnv import HandDemoEnv
from domain_object.builder import DomainObject


class RobotEnvFactory:
    @staticmethod
    def create(env_name: str, domain_object: DomainObject):

        if "grasp" in env_name : return ManipulatorGraspEnv(domain_object)
        if "hand"  in env_name : return HandDemoEnv(domain_object)

        raise NotImplementedError()
