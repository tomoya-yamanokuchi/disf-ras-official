import mujoco
import mujoco.viewer
import numpy as np
from .gui_context_setting import gui_context_setting
from .Camera import Camera


class MyViewerWrapper:
    def __init__(self,
            model,
            data,
            camera : Camera,
            opt,
            scn,
            config_env
        ):
        """
        - use_gui: Trueならmujoco.viewerを利用してウィンドウ表示，
                   Falseならオフスクリーン描画を使う
        - env    : PandaArmEnv などMuJoCoモデル＋dataを持っている環境
        """
        self.model  = model
        self.data   = data
        self.camera = camera
        self.opt    = opt
        self.scn    = scn
        # ---
        self.config_env  = config_env
        self.use_gui     = config_env.viewer.use_gui
        self.width       = config_env.renderer.width
        self.height      = config_env.renderer.height
        # ---
        # GUIモードの場合に使うviewerハンドル
        self._gui_viewer = None
        self._gui_context_manager = None

        # オフスクリーンの場合に使うMjrContext等
        self._offscreen_context = None
        self._render_buf = None

        # フレーム（画像）を貯めるリスト
        self.frames = []



    def launch(self):
        """GUIの場合は launch_passive, オフスクリーンの場合は MjrContext を作る。"""
        if self.use_gui:
            # GUI表示モード
            viewer_params = {
                "model"        : self.model,
                "data"         : self.data,
                "show_left_ui" : self.config_env.viewer.show_ui,
                "show_right_ui": self.config_env.viewer.show_ui,
            }
            self._gui_context_manager = mujoco.viewer.launch_passive(**viewer_params)
        else:
            # オフスクリーンモード
            # Renderer のインスタンス化 (必要)
            self._renderer = mujoco.Renderer(
                model  = self.model,
                height = self.config_env.renderer.height,
                width  = self.config_env.renderer.width,
            )
            self._offscreen_context = mujoco.MjrContext(
                self.model, mujoco.mjtFontScale.mjFONTSCALE_100
            )
            self._render_buf = np.zeros((self.height, self.width, 3), dtype=np.uint8)

            # import ipdb; ipdb.set_trace()


    def __enter__(self):
        """
        with構文で使われる場合を想定。
        GUIモードなら launch_passive の返す viewer ハンドルを受け取る。
        """
        if self.use_gui and self._gui_context_manager is not None:
            self._gui_viewer = self._gui_context_manager.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """withブロックを抜けたらクリーンアップ"""
        if self.use_gui and self._gui_context_manager is not None:
            self._gui_context_manager.__exit__(exc_type, exc_val, exc_tb)

    def initialize_for_env(self):
        """
        環境側が持つ Viewer 初期化処理があれば呼ぶ（例: env.mujoco_viewer.initialize(viewer)）。
        """
        # ---- contact visualization options ----
        self.model.vis.scale.contactwidth  = self.config_env.viewer.scale.contactwidth  # contact forceベクトルの太さ
        self.model.vis.scale.contactheight = self.config_env.viewer.scale.contactheight # contact forceベクトルの矢印先端(ヘッド)の長さ
        self.model.vis.scale.forcewidth    = self.config_env.viewer.scale.forcewidth    # contact forceをどれくらいのスケールで可視化するか
        # import ipdb; ipdb.set_trace()
        gui_context_setting(self._gui_context_manager, config_viewer=self.config_env.viewer)


    def sync(self):
        """
        いままでの `viewer.sync()` を置き換えるメソッド。
        - GUIモードなら実際に `_gui_viewer.sync()` を呼ぶ
        - オフスクリーンなら「フレーム取得して保存」等にする
        """
        if self.use_gui:
            if self._gui_viewer is not None:
                self._gui_viewer.sync()
        else:
            # オフスクリーンの場合: 今までは sync() で何をしていたか次第ですが、
            # 「ここでフレームをキャプチャする」「何もしない」などアレンジ可能。

            # # import ipdb; ipdb.set_trace()
            # if new_lookat is not None:
            #     self.cam.lookat[:] = new_lookat[:]
            # else:
            #     self.cam.lookat[:] = self.config_env.camera.lookat[:]

            return self._capture_frame()

    def render(self):
        """
        任意で `viewer.render()` 相当の機能を提供したい場合。
        GUIなら `_gui_viewer.render()` (MuJoCo 2.3+でピクセル取得可能)
        オフスクリーンなら mjr_render + mjr_readPixels().
        """
        if self.use_gui:
            if self._gui_viewer is not None:
                frame = self._gui_viewer.render()  # 返り値がフレーム配列 (height, width, 3)
                return frame
            return None
        else:
            return self._capture_frame()


    def _capture_frame(self):
        if self._offscreen_context is None:
            return None
        # -----------------------------------

        # シーンを更新
        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.opt,          # ここがMjvOption
            None,            # perturb使わないならNone
            self.camera.cam,             # MjvCamera
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scn
        )

        # ビューポート設定
        viewport = mujoco.MjrRect(0, 0, self.width, self.height)

        # レンダリング
        mujoco.mjr_render(viewport, self.scn, self._offscreen_context)

        # ピクセル読み取り
        mujoco.mjr_readPixels(
            rgb      = self._render_buf,
            depth    = None,
            viewport = viewport,
            con      = self._offscreen_context
        )


        # OpenGLの描画は上下が反転している場合があるので flipud
        frame = np.flipud(self._render_buf.copy())

        # framesに保存
        self.frames.append(frame)

        return frame


