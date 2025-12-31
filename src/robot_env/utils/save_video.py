import imageio
from print_color import print


def save_video(
        frames    : list,
        fps       : int = 30,
        skip      : int = 1,
        save_path : str ="./output.mp4",
    ):
    """
    :param frames: すべてのフレーム (list of ndarray, shape=(H,W,3) など)
    :param save_path: 出力する動画ファイルのパス
    :param fps: 元々のフレームレート (skip=1時の想定)
    :param skip: 何枚おきにフレームを使うか（例: skip=5なら5フレームに1度だけ保存）
    """
    # ---------------------------------------------------------------
    if len(frames) == 0:
        print(f"no frames are saved", tag = 'save_vide', tag_color='yellow', color='yellow')
        return
    # ---------------------------------------------------------------


    # N 枚おき (skip 枚おき) のフレームだけを取り出す
    frames_to_save = frames[::skip]

    # 再生速度（fps）はどうする？
    # もし「スキップしても再生速度は変えずに高速再生にしたい」なら fps はそのまま。
    # 「スキップしても元の動作時間を維持したい（再生速度を遅くする）」なら、fps を調整 (fps / skip)。
    # ここでは「間引いても再生速度は上がる」(= fps はそのまま) とします。
    new_fps = fps

    print(f"Original frames: {len(frames)}, saving {len(frames_to_save)} frames.")
    print(f"Using FPS={new_fps}")

    if not frames_to_save:
        print("No frames to save (frames_to_save is empty).")
        return

    with imageio.get_writer(save_path, fps=new_fps) as writer:
        for f in frames_to_save:
            writer.append_data(f)
    print(f"Saved '{save_path}' with {len(frames_to_save)} frames.")
