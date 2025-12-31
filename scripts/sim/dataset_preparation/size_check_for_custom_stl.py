import trimesh
from pathlib import Path

if __name__ == '__main__':
    stl_path = Path("/home/cudagl/disf_ras/models/custom_mesh/real_experiment_object/T/T.stl")

    mesh_tm = trimesh.load(stl_path, force="mesh")

    print("bounds:", mesh_tm.bounds)    # [[xmin, ymin, zmin], [xmax, ymax, zmax]]
    print("extents:", mesh_tm.extents)  # [dx, dy, dz]
    print("scale (max extent):", mesh_tm.extents.max())
