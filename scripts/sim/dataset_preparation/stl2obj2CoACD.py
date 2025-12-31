import os
import subprocess
import trimesh
from pathlib import Path
from args import parse_args

class STL2OBJ2CoACD:
    def __init__(self, stl_root: str, stl_name: str):
        self.stl_root = stl_root
        self.stl_name = stl_name
        # self.obj_dir  = os.path.join(stl_root, stl_name)
        self.stl_path = os.path.join(self.stl_root, f"{stl_name}.stl")
        self.obj_path = os.path.join(self.stl_root, f"{stl_name}.obj")
        self.CoACD_obj_out_path = self.stl_root

    def convert(self, resolution: int = 50, density: int = 100):


        mesh_tm = trimesh.load(self.stl_path, force="mesh")

        # 必要ならここでスケール調整 (例: mm -> m)
        mesh_tm.apply_scale(0.001)

        # 2. OBJ に書き出し（obj2mjcf が読む用の "元メッシュ"）
        mesh_tm.export(self.obj_path)

        # 3. obj2mjcf を YCB と同じノリで実行
        cmd = [
            "obj2mjcf",
            "--obj-dir", str(self.CoACD_obj_out_path),
            "--obj-filter", rf"{stl_name}\.obj",  # ディレクトリに他の obj があってもこの1つだけ処理したい場合
            "--save-mjcf",
            "--decompose",
            "--overwrite",
            "--add-free-joint",
            "--coacd-args.preprocess-resolution", str(resolution),
            "--density", str(density),
        ]

        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)



if __name__ == '__main__':
    args = parse_args()
    # -----
    stl_root = Path(f"/home/cudagl/disf_ras/models/custom_mesh/{args.object_name}")
    stl_name = "textured"
    # -----
    stl2obj2CoACD = STL2OBJ2CoACD(stl_root=stl_root, stl_name=stl_name)
    stl2obj2CoACD.convert(resolution=50, density=args.density)
