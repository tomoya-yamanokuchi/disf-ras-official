from mujoco_grasping import ISFMujocoGraspingYCBObjectEvaluation
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
        object_name_list = args.object_name.split(","),
        isf_model        = args.method
    )


'''
example:
    usr/local/bin/python /home/cudagl/disf_ras/scripts/grasp/sim/grasp_one_object.py --robot_name ur5e --object_name custom_Hammer --method visf
'''


