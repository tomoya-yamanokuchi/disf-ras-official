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

    # ========================= box_composite_mass_sweep =========================
    p.add_argument("--total_mass_list", type = str, default = "0.50")
    p.add_argument("--condition_list" , type = str, default = None)

    p.add_argument("--condition"       , type = str, default = "uniform")
    p.add_argument("--total_mass"      , type = float, default = 0.5)
    # --- weight bias ---
    p.add_argument("--bias_mild"       , type = float, default = 0.5)
    p.add_argument("--bias_medium"     , type = float, default = 1.0)
    p.add_argument("--bias_large"      , type = float, default = 1.5)
    # --- grasp position ---
    p.add_argument("--grasp_x"         , type = float, default = -0.08)
    # --- box size ---
    p.add_argument("--size_x"          , type = float, default = 0.20)
    p.add_argument("--size_y"          , type = float, default = 0.04)
    p.add_argument("--size_z"          , type = float, default = 0.04)
    # --- trial setting ---
    p.add_argument("--n_trials"        , type = int  , default = 10)
    p.add_argument("--seed"            , type = int  , default = 0)
    p.add_argument("--max_angle_deg"   , type = float, default = 5.0)

    p.add_argument("--object_root_dir" , type = str  , default = "./models/box_mesh")
    p.add_argument("--results_save_dir", type = str  , default = "/home/cudagl/dataset/RAS_results/box5")
    return p.parse_args()


