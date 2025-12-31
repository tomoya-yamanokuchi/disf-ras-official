import open3d as o3d





if __name__ == '__main__':

    fname = "/home/cudagl/disf_ras/models/custom_mesh/real_experiment_object/OldCamera/textured.stl"

    mesh = o3d.read_triangle_mesh(fname)
    PDS  = mesh.sample_points_poisson_disk(number_of_point=500)

