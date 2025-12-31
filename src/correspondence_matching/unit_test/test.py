
from correspondence_matching import ICPPointMatcher
from correspondence_matching import ICPPointFiltering



if __name__ == '__main__':
    import numpy as np
    source = np.random.rand(10, 3)  # Random 10 points in 3D
    target = np.random.rand(20, 3)  # Random 20 points in 3D

    # -----
    matcher                    = ICPPointMatcher()
    correspondences, distances = matcher.find_correspondences(source, target)

    filter = ICPPointFiltering()
    icp_filter_result = filter.filter_duplicate_correspondences(
        source, correspondences, distances
    )

    print("         source.shape = ", source.shape)
    print("correspondences.shape = ", icp_filter_result.correspondences.shape)
    print("filtered_source.shape = ", icp_filter_result.distances.shape)

    # import ipdb ; ipdb.set_trace()
