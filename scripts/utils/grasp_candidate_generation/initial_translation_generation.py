from domain_object.builder import SelfContainedDomainObjectBuilder, DomainObject
from domain_object.director.isf_planning.InitialTranslationGenerationDirector import InitialTranslationGenerationDirector
from args import parse_args
import numpy as np
from point_cloud_clustering.visualize_point_clusters import visualize_point_clusters
from service import print_cluster_centers_pretty


def run(robot_name, object_name, N_CLUSTERS, save, save_path, figsize):
    builder       = SelfContainedDomainObjectBuilder()
    director      = InitialTranslationGenerationDirector()
    domain_object = director.construct(
        builder         = builder,
        robot_name      = robot_name,
        object_name     = object_name,
        isf_model       = "disf",
        N_CLUSTERS      = N_CLUSTERS,
    )
    object_kmeans_centers = domain_object.centers
    object_kmeans_labels  = domain_object.labels

    print("--------------------------")
    print_cluster_centers_pretty(object_kmeans_centers)

    # visualize
    points = np.array(domain_object.object_whole_surface.points)  # (907, 3)
    visualize_point_clusters(
        points            = points,
        labels            = object_kmeans_labels,
        centers= object_kmeans_centers,
        show_axis_numbers = False,
        save              = save,
        save_path         = save_path,
        point_size        = 5,
        figsize           = figsize
    )

if __name__ == '__main__':
    import os
    args = parse_args()

    figsize = (13, 13)

    save_dir = "/home/cudagl/data/k-means_result/"
    save_path = os.path.join(save_dir, f"{args.object_name}_ncluster={str(args.n_cluster)}.png")

    run(robot_name  = args.robot_name,
        object_name = args.object_name,
        # ---
        N_CLUSTERS  = args.n_cluster,
        save        = args.save,
        save_path   = save_path,
        figsize     = figsize
    )
