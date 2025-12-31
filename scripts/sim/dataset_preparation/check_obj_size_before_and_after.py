import trimesh
from pathlib import Path

root = Path("/home/cudagl/disf_ras/models/custom_mesh/real_experiment_object/OldCamera")

mesh_vis = trimesh.load(root  / "textured" / "textured.obj", force="mesh")
mesh_col0 = trimesh.load(root / "textured"  / "textured_collision_0.obj", force="mesh")
mesh_col1 = trimesh.load(root / "textured"  / "textured_collision_1.obj", force="mesh")


print("-------------- Visual Mesh and Collision Meshes ----------------")
print("textured.obj bounds:", mesh_vis.bounds)
print("textured_collision_0 bounds:", mesh_col0.bounds)
print("textured_collision_1 bounds:", mesh_col1.bounds)

# print("-------------- Visual Mesh and Collision Meshes ----------------")
# print("bounds:", mesh_tm.bounds)    # [[xmin, ymin, zmin], [xmax, ymax, zmax]]

print("extents0:", mesh_col0.extents)  # [dx, dy, dz]
print("extents1:", mesh_col1.extents)  # [dx, dy, dz]


# print("scale (max extent):", mesh_tm.extents.max())
# print("scale (max extent):", mesh_tm.extents.max())
