#!/bin/bash

# 処理するオブジェクトのリスト
objects=(

    # "001_chips_can"
    # ---- Food items ----

    # "003_cracker_box"
    "006_mustard_bottle"
    # "007_tuna_fish_can"
    # "011_banana"
    # "012_strawberry"
    # "016_pear"
    # "018_plum"
    # ---- Kitchen items ----
    # "019_pitcher_base"
    # "021_bleach_cleanser"
    # "025_mug"
    # "029_plate"
    # "033_spatula"
    # # ---- Tool items ----
    # "035_power_drill"
    # "037_scissors"
    # "038_padlock"
    # "042_adjustable_wrench"
    # "043_phillips_screwdriver"
    # "048_hammer"
    "052_extra_large_clamp"
    # "049_small_clamp"
)

# 各オブジェクトに対して `obj2mjcf` を実行
for obj in "${objects[@]}"; do
    echo "Processing $obj..."
    obj2mjcf --obj-dir ~/disf_ras/models/ycb/"$obj"/tsdf --save-mjcf --decompose --overwrite --add-free-joint --coacd-args.preprocess-resolution 50
done

echo "All objects processed."
