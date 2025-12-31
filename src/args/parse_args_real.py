import argparse


def parse_args_real():
    p = argparse.ArgumentParser()

    # ------------ common ------------
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

    # ---------- real specific ----------
    p.add_argument("--host", type=str, default="192.168.11.7", help="UR controller IP")
    p.add_argument("--duration", type=float, default=1.0, help="averaging window [s]")
    p.add_argument("--hz", type=float, default=125.0, help="sampling rate for averaging")
    p.add_argument(
        "--out_dir",
        type=str,
        default="./calibration_data",
        help="output directory (default: current dir)",
    )
    p.add_argument(
        "--no_home",
        action="store_true",
        help="do not move to home (use current pose)",
    )
    p.add_argument(
        "--result_save_root",
        default="../data/real_experiment",
    )
    p.add_argument(
        "--abort_mult",
        type    = float,
        default = 3.0,
    )
    return p.parse_args()
