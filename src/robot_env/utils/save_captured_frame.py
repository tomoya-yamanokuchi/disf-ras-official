import os
import numpy as np
from PIL import Image
from print_color import print


def save_captured_frame(
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
