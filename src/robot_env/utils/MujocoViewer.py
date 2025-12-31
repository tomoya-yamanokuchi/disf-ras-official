import mujoco
from mujoco.viewer import Handle


class MujocoViewer:
    def __init__(self, model, data, config):
        self.model  = model
        self.data   = data
        self.config = config

    def initialize(self, viewer: Handle):
        if not isinstance(viewer, Handle):
            return
        # ---- geom render setting ----
        viewer.opt.geomgroup[1] = self.config.viewer.opt.geomgroup.group_1
        viewer.opt.geomgroup[2] = self.config.viewer.opt.geomgroup.group_2
        viewer.opt.geomgroup[3] = self.config.viewer.opt.geomgroup.group_3
        # ---- site render setting ----
        viewer.opt.sitegroup[4] = self.config.viewer.opt.sitegroup.group_4
        # ---- Render contact position ----
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = self.config.viewer.opt.contact_point
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = self.config.viewer.opt.contact_force
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL]   = self.config.viewer.opt.convex_full
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT]  = self.config.viewer.opt.transparent
        # ---- Render visualization option ----
        viewer.opt.frame = self.config.viewer.opt.frame
        viewer.opt.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        # ---- contact visualization options ----
        self.model.vis.scale.contactwidth  = self.config.viewer.scale.contactwidth  # contact forceベクトルの太さ
        self.model.vis.scale.contactheight = self.config.viewer.scale.contactheight # contact forceベクトルの矢印先端(ヘッド)の長さ
        self.model.vis.scale.forcewidth    = self.config.viewer.scale.forcewidth    # contact forceをどれくらいのスケールで可視化するか


    def get_viewer_params(self, show_ui: bool = None):
        return {
            "model"        : self.model,
            "data"         : self.data,
            "show_left_ui" : self.config.viewer.show_ui if show_ui is None else show_ui,
            "show_right_ui": self.config.viewer.show_ui if show_ui is None else show_ui,
        }
