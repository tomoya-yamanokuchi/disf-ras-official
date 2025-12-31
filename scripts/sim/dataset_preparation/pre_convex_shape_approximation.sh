#!/bin/bash

# 処理するオブジェクトのリスト
objects=(
    # ----- YCB object -----
    "006_mustard_bottle"     # 1
    "011_banana"             # 2
    "029_plate"              # 3
    "033_spatula"            # 4
    "035_power_drill"        # 5
    "037_scissors"           # 6
    "042_adjustable_wrench"  # 7
    "052_extra_large_clamp"  # 8
    "058_golf_ball"          # 9
    "065-j_cups"             # 10
)

# 各オブジェクトに対して `obj2mjcf` を実行
for obj in "${objects[@]}"; do
    echo "Processing $obj..."
    obj2mjcf --obj-dir ~/disf_ras/models/ycb/"$obj"/tsdf --save-mjcf --decompose --overwrite --add-free-joint --coacd-args.preprocess-resolution 50
done

echo "All objects processed."
