from domain_object.builder import DomainObject
from robot_env.instance.arm.ManipulatorGraspEnv import ManipulatorGraspEnv
from print_color import print


class TimeWatch:
    """シミュレーション時間を管理するユーティリティ"""
    def __init__(self, domain_object: DomainObject):
        self.env : ManipulatorGraspEnv = domain_object.env
        self.TIMEOUT           = domain_object.config_ik_solver.TIMEOUT
        self.start_time        = None

    def start(self):
        self.start_time = self.env.time()

    def check_continue(self) -> bool:
        current_time = self.env.time()
        if (current_time - self.start_time) >= self.TIMEOUT:
            print(f"Timeout! Could not reach the target within {self.TIMEOUT:.2f} seconds.",
                  tag = "TimeWatch", color="m", tag_color="m")
            return False
        else:
            return True


