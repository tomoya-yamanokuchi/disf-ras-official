import mujoco
import numpy as np

class Camera:
    def __init__(self, config, model):
        self.config = config
        self.cam    = mujoco.MjvCamera()
        if config.camera.default_free_camera :
            mujoco.mjv_defaultFreeCamera(model, self.camera)
        # ---
        # if config.camera.lookat   : self.camera.lookat[:] = config.camera.lookat
        # if config.camera.distance : self.camera.distance  = config.camera.distance
        # if config.camera.azimuth  : self.camera.azimuth   = config.camera.azimuth
        # if config.camera.elevation: self.camera.elevation = config.camera.elevation
        # ---

    def set_zoom_with_fingertip_center(self, fingertip_center: np.ndarray):
        self.cam.lookat[:] = fingertip_center[:]
        self.cam.distance  = self.config.camera.zoom.distance
        self.cam.azimuth   = self.config.camera.zoom.azimuth
        self.cam.elevation = self.config.camera.zoom.elevation

    def set_overview(self):
        self.cam.lookat[:] = self.config.camera.overview.lookat
        self.cam.distance  = self.config.camera.overview.distance
        self.cam.azimuth   = self.config.camera.overview.azimuth
        self.cam.elevation = self.config.camera.overview.elevation
