import mujoco


class Scene:
    def __init__(self, config, model):
        self.scene = mujoco.MjvScene(model, maxgeom=2000)

        # たとえば contact point, contact force, convex hull, transparent など
        # self.scene.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1 if config.viewer.opt.contact_point  else 0
        # self.scene.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1 if config.viewer.opt.contact_force  else 0
        # self.scene.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL]   = 1 if config.viewer.opt.convex_full    else 0
        # self.scene.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT]  = 1 if config.viewer.opt.transparent   else 0
