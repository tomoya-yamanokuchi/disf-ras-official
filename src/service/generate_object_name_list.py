


def generate_object_name_list():
    return [
        # "box",
        # =================== simulation YCB Object ===================
        # ---- Food items (panda) ----
        # "001_chips_can",      # (1) 0.341 [kg] -> fail: dynamics
        # "003_cracker_box",    # (2) 0.508 [kg] -> success: ok
        # "006_mustard_bottle", # (3) 0.181 [kg] -> success: ok
        "011_banana",         # (4) 0.061 [kg] -> success: ok
        # "012_strawberry",     # (5) 0.008 [kg] -> success: ok
        # "016_pear",           # (6) 0.028 [kg] -> fail: pre-grasp relation ship
        # "018_plum",           # (7) 0.011 [kg] -> success: ok
        # ---- Kitchen items (kuka) ----
        # "019_pitcher_base",     # (1) 0.564 [kg] -> fail: size too large
        # "021_bleach_cleanser",
        # "022_windex_bottle",
        # "023_wine_glass",
        # "025_mug",
        # "029_plate",
        # "033_spatula",

        # =================== Custom Real 3D Object ===================
        # "custom_T",
        # "custom_Hammer",
        # "custom_OldCamera",
    ]
