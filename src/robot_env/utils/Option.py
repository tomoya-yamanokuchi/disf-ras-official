import mujoco
import re


class Option:
    def __init__(self, config_viewer):
        self.option = mujoco.MjvOption()
        mujoco.mjv_defaultOption(self.option)
        # ------------------------------------------------------
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
            self.option.geomgroup[int(number)] = val

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
            self.option.sitegroup[int(number)] = val
        print(f"----------------------------------------------")

        # -- frame (mjFRAME_NONE, mjFRAME_BODY, mjFRAME_WORLD etc.)
        # もともと viewer.opt.frame に入っていた値に応じて設定
        self.option.frame = config_viewer.opt.frame  # 数値 or enum になる場合は注意
        # -----
        self.option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1 if config_viewer.opt.contact_point  else 0
        self.option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1 if config_viewer.opt.contact_force  else 0
        self.option.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL]   = 1 if config_viewer.opt.convex_full    else 0
        self.option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT]  = 1 if config_viewer.opt.transparent   else 0

