from mujoco_grasping import ISFMujocoGraspingYCBObjectEvaluation
from service import generate_object_name_list
from args import parse_args


def run(robot_name, object_name_list, isf_model):
    grasp = ISFMujocoGraspingYCBObjectEvaluation()
    grasp.evaluate(
        robot_name       = robot_name,
        object_name_list = object_name_list,
        isf_model        = isf_model,
    )


if __name__ == '__main__':
    args = parse_args()
    run(
        robot_name       = args.robot_name,
        object_name_list = generate_object_name_list(),
        isf_model        = args.method
    )
