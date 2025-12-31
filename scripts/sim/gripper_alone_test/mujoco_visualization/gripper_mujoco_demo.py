import os
import pathlib
from domain_object.director.mujoco import MujocoHandEnvDirector
from domain_object.builder import SelfContainedDomainObjectBuilder
from mujoco_grasping.MujocoHandAloneEnv import MujocoHandAloneEnv
from args import parse_args


def run(robot_name: str = "panda"):
    # ------- set hand xml path -------
    cwd_path = pathlib.Path(os.getcwd())
    load_xml_path = cwd_path.joinpath(
        f"./assets/{robot_name}/{robot_name}_hand.xml")
    # ---------------------------------
    builder       = SelfContainedDomainObjectBuilder()
    director      = MujocoHandEnvDirector()
    domain_object = director.construct(
        builder         = builder,
        robot_name      = robot_name,
        robot_mode      = "hand",
        load_xml_path   = load_xml_path,
    )
    # ---------------------------------
    grasp = MujocoHandAloneEnv(domain_object)
    grasp.execute()


if __name__ == '__main__':
    args = parse_args()
    run(robot_name = args.robot_name)
