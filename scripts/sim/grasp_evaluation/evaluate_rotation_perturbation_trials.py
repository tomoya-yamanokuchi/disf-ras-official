from mujoco_grasping import ISFMujocoGraspingWithPosePerturbation
from args import parse_args


def run(
    robot_name,
    object_name_list,
    isf_model,
    n_trials      ,
    seed         ,
    max_angle_deg,
):
    grasp = ISFMujocoGraspingWithPosePerturbation()
    results = grasp.evaluate(
        robot_name       = robot_name,
        object_name_list = object_name_list,
        isf_model        = isf_model,
        n_trials         = n_trials,
        seed             = seed,
        max_angle_deg    = max_angle_deg,
    )

    import ipdb; ipdb.set_trace()

    return results


if __name__ == "__main__":
    args = parse_args()
    run(
        robot_name       = args.robot_name,
        object_name_list = args.object_name.split(","),
        isf_model        = args.method,
        n_trials         = args.n_trials,
        seed             = args.seed,
        max_angle_deg    = args.max_angle_deg,
    )
