import itertools
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).parent / "grasp_one_object.py"
PYTHON = "python"  # 環境に合わせて変更


# robots  = ["ur5e", "panda", "kuka"]
# methods = ["disf", "visf", "cma"]

robots  = ["ur5e"]
methods = ["disf", "visf", "cma"]

objects = [
    # # --- YCB objects ---
    # "006_mustard_bottle",
    # "011_banana",
    # "029_plate",
    # "033_spatula",
    # "035_power_drill",
    # "037_scissors",
    # "042_adjustable_wrench",
    # "052_extra_large_clamp",
    # "058_golf_ball",
    # "065-j_cups",
    # # --- Custom real 3D objects ---
    # "custom_T",
    # "custom_RubberDuck",
    # "custom_Hammer",
    # "custom_WineGlass",
    # "custom_OldCamera",
    # --- Observed daily objects ---
    "observed_T",
    "observed_RubberDuck",
    "observed_Hammer",
    "observed_WineGlass",
    "observed_OldCamera",
    # ---
    "observed_Tripod",
    "observed_USB",
    "observed_Controller",
    "observed_Tape",
]

for robot, method, obj in itertools.product(robots, methods, objects):
    print(f"Running robot={robot}, object={obj}, method={method}")
    cmd = [
        PYTHON,
        str(SCRIPT),
        "--robot_name", robot,
        "--object_name", obj,
        "--method", method,
    ]
    subprocess.run(cmd, check=True)
