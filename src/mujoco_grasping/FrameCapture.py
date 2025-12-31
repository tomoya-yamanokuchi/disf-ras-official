import os
import numpy as np
from PIL import Image
from print_color import print
from domain_object.builder import DomainObject


class FrameCapture:
    def __init__(self, domain_object: DomainObject):
        self.use_gui          = domain_object.config_env.viewer.use_gui
        self.results_save_dir = domain_object.results_save_dir
        self.model_name       = domain_object.model_name


    def home(self, frame: np.ndarray):
        self.save(frame, pose_name="home")

    def pregrasp(self, frame: np.ndarray):
        self.save(frame, pose_name="pregrasp")

    def postgrasp(self, frame: np.ndarray):
        self.save(frame, pose_name="postgrasp")

    def grasp(self, frame: np.ndarray):
        self.save(frame, pose_name="grasp")

    def save(self, frame: np.ndarray, pose_name: str):
        if self.use_gui:
            return
        save_path = os.path.join(self.results_save_dir, pose_name + f"_{self.model_name}" +".png")
        self._save(frame, save_path)


    def _save(self,
            frame     : np.ndarray,
            save_path : str ="./output.png",
        ):

        # frameはnumpy配列(OpenGLのRGBバッファ)なので、PIL Imageに変換
        img = Image.fromarray(frame.astype(np.uint8))

        # 保存先のディレクトリが存在しない場合は作成
        dir_name = os.path.dirname(save_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)

        # 画像ファイルとして保存 (PNG)
        img.save(save_path)
        print(f"Saved frame to {save_path}", tag = 'save_captured_frame', tag_color='yellow', color='yellow')



