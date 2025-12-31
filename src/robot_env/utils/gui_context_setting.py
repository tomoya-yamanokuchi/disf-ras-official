import re
import mujoco
from mujoco.viewer import Handle
from omegaconf import DictConfig
from print_color import print


def gui_context_setting(
        _gui_context_manager: Handle,
        config_viewer : DictConfig,
    ):
    if _gui_context_manager is None:
        print(f"no gui_context_setting is setted", tag = 'gui_context_setting', tag_color='yellow', color='yellow')
        return
    # ----------------------------------------------------------------------------------------
    pattern = r'\d+'
    print(f"----------------------------------------------")
    # --------------- geomgroup setting ---------------
    for key, val in dict(config_viewer.opt.group_enable.geomgroup).items():
        match = re.search(pattern, key)
        # ----
        if not match:
            print("数字が見つかりませんでした。")
            raise NotImplementedError()
        # ----
        number = match.group()
        print(f" group_enable: geomgroup[{int(number)}] = {val}")
        _gui_context_manager.opt.geomgroup[int(number)] = val

    # --------------- sitegroup setting ---------------
    for key, val in dict(config_viewer.opt.group_enable.sitegroup).items():
        match = re.search(pattern, key)
        # ----
        if not match:
            print("数字が見つかりませんでした。")
            raise NotImplementedError()
        # ----
        number = match.group()
        print(f" group_enable: sitegroup[{int(number)}] = {val}")
        _gui_context_manager.opt.sitegroup[int(number)] = val
    print(f"----------------------------------------------")

    # --------------- sitegroup setting ---------------
    # ---- Render contact position ----
    _gui_context_manager.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = config_viewer.opt.contact_point
    _gui_context_manager.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = config_viewer.opt.contact_force
    _gui_context_manager.opt.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL]   = config_viewer.opt.convex_full
    _gui_context_manager.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT]  = config_viewer.opt.transparent
    # ---- Render visualization option ----
    _gui_context_manager.opt.label = config_viewer.opt.label
    _gui_context_manager.opt.frame = config_viewer.opt.frame
    _gui_context_manager.opt.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False

