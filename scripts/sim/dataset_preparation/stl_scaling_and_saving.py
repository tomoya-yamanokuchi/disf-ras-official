import os
import open3d as o3d


def stl_scaling_and_saving(input_stl: str, output_stl: str, scale_factor: float):
    # 1. STLメッシュを読み込み
    mesh = o3d.io.read_triangle_mesh(input_stl)
    if not mesh.has_vertices():
        raise RuntimeError("メッシュが空です。読み込みに失敗している可能性があります。")

    # 2. 必要に応じて原点周りにスケーリングするかどうか検討
    #   Open3Dの mesh.scale は、デフォルトだと原点 (0,0,0) を中心にスケールします。
    #   メッシュを重心中心でスケールしたい場合は下記のように center を指定します。

    center = mesh.get_center()   # メッシュ頂点の重心

    mesh_scaled = mesh.scale(scale_factor, center=center)

    # 3. スケール後のSTLを保存
    o3d.io.write_triangle_mesh(output_stl, mesh_scaled)
    print(f"Scaled STL saved to: {output_stl}")


if __name__ == '__main__':
    # --- パラメータ ---
    stl_dir = "/home/cudagl/disf_ras/models/custom_mesh/complex"
    # ---
    input_stl  = "Hammer.stl"
    output_stl = f"scaled_{input_stl}"    # スケール後のSTL
    scale_factor = 0.7                 # 70% に縮小する
    # ---
    stl_dir_path = os.path.join(stl_dir, "input_stl")

    stl_scaling_and_saving(input_stl, output_stl, scale_factor)
