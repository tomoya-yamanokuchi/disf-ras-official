import mujoco
import numpy as np


class Renderer:
    def __init__(self, config, model, data, camera):
        self.data     = data
        self.camera   = camera
        self.renderer = mujoco.Renderer(
            model  = model,
            height = config.renderer.height,
            width  = config.renderer.width,
        )

    def update_scene(self):
        self.renderer.update_scene(self.data, self.camera)

    def render(self, bgr: bool=True):
        rgb = self.renderer.render()
        rgb = rgb.astype(float) / 255.
        rgb = (rgb * 255).astype(np.uint8)
        # ---
        # import ipdb ; ipdb.set_trace()
        if bgr: return rgb[..., ::-1]
        else  : return rgb

    def close(self):
        self.renderer.close()