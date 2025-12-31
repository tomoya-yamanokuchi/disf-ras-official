import os
from copy import deepcopy
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra


class ConfigLoader:
    @staticmethod
    def load_env(config_name: str):
        if config_name is None:
            config_name = "config"
        GlobalHydra.instance().clear()
        initialize(config_path="./config/env")
        return compose(config_name=config_name)

    @staticmethod
    def load_ik_solver(config_name: str):
        if config_name is None:
            config_name = "default_ik_solver"
        GlobalHydra.instance().clear()
        initialize(config_path="./config/ik_solver")
        return compose(config_name=config_name)

    @staticmethod
    def load_cma(config_name: str):
        GlobalHydra.instance().clear()
        initialize(config_path="./config/cma")
        return compose(config_name=config_name)

    @staticmethod
    def load_grasp_evaluation(config_name: str):
        GlobalHydra.instance().clear()
        initialize(config_path="./config/grasp_evaluation")
        return compose(config_name=config_name)

    @staticmethod
    def load_icp(config_name: str):
        GlobalHydra.instance().clear()
        initialize(config_path="./config/icp")
        return compose(config_name=config_name)

    @staticmethod
    def load_point_cloud_data(config_name: str):
        if config_name is None:
            config_name = "config"
        GlobalHydra.instance().clear()
        initialize(config_path="./config/point_cloud_data")
        return compose(config_name=config_name)

    @staticmethod
    def load_isf(config_name: str):
        if config_name is None:
            config_name = "config"
        GlobalHydra.instance().clear()
        initialize(config_path="./config/isf")
        return compose(config_name=config_name)

    @staticmethod
    def load_gripper_surface(config_name: str):
        GlobalHydra.instance().clear()
        initialize(config_path="./config/gripper_surface")
        return compose(config_name=config_name)
