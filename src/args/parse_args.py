import argparse


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--robot_name",
        type    = str,
        default = "panda",
        choices = ["panda", "ur5e", "kuka"],
        help    = "please select one from (panda, ur5e, kuka)")

    p.add_argument("--object_name",
        type    = str,
        default = "006_mustard_bottle",
        help    = "YCB object name, e.g., 006_mustard_bottle")

    p.add_argument("--method",
        type    = str,
        default = "visf",
        choices = ["cma", "visf", "disf"],
        help    = "please select one from (cma, visf, dis)")

    p.add_argument("--n_cluster",
        type    = int,
        default = 10,
        help    = "please give the int number information (e.g. 5, 8, 10)")

    p.add_argument("--density",
        type    = int,
        default = 200,
        help    = "please give the int number information (e.g. 100, 200)")

    p.add_argument(
        "--save",
        action="store_true",
        help="Save results to disk."
    )


    return p.parse_args()
