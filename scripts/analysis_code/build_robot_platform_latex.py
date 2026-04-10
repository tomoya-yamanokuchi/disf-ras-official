import pandas as pd

CSV_PATH = "/home/cudagl/data/RAS_results/summary_grasp_results.csv"  # パスは適宜変更



def build_robot_platform_latex(csv_path: str) -> None:
    df = pd.read_csv(csv_path)

    # --- Setting を object から推定 ---
    known_objects = [
        "custom_T",
        "custom_RubberDuck",
        "custom_Hammer",
        "custom_WineGlass",
        "custom_OldCamera",
    ]
    observed_objects = [
        "006_mustard_bottle",
        "011_banana",
        "029_plate",
        "033_spatula",
        "035_power_drill",
        "037_scissors",
        "042_adjustable_wrench",
        "052_extra_large_clamp",
        "058_golf_ball",
        "065-j_cups",
    ]

    def infer_setting(obj: str) -> str:
        if obj in known_objects:
            return "Known-shape"
        elif obj in observed_objects:
            return "Observed-shape"
        else:
            return "Unknown"

    df = df.copy()
    df["setting"] = df["object"].apply(infer_setting)
    df = df[df["setting"] != "Unknown"]  # 念のため Unknown は除外

    # --- robot / method を表示名にマップ ---
    robot_map = {"panda": "Panda", "ur5e": "UR5e", "kuka": "iiwa"}
    method_map = {"cma": "CMA-ES", "visf": "VISF", "disf": "DISF"}

    df["robot_disp"] = df["robot"].map(robot_map)
    df["method_disp"] = df["method"].map(method_map)

    # --- setting × robot × method ごとの成功率（オブジェクト平均） ---
    # success_rate は 0〜1 の値なので、ここではそのまま平均しておく
    grouped = (
        df.groupby(["setting", "robot_disp", "method_disp"])["success_rate"]
        .mean()
    )

    setting_order = ["Known-shape", "Observed-shape"]
    robot_order = ["Panda", "UR5e", "iiwa"]
    method_order = ["CMA-ES", "VISF", "DISF"]

    # ピボットして (setting, robot) × method の表に
    pivot = (
        grouped.unstack("method_disp")
        .reindex(
            index=pd.MultiIndex.from_product(
                [setting_order, robot_order],
                names=["setting", "robot_disp"],
            )
        )
        .reindex(columns=method_order)
    )

    # Setting ごとの「Average over robots」
    avg_over_robots = pivot.groupby(level="setting").mean().reindex(setting_order)

    # 全体の Total Average（両 Setting × 全 robot）
    total_avg = pivot.mean(axis=0)

    # --- LaTeX 行の整形（各行で最大値を太字に） ---
    def format_row(values: pd.Series, bold_best: bool = True):
        vals = []
        max_val = values.max() if bold_best else None
        for m in method_order:
            v = values[m]
            if pd.isna(v):
                s = "-"
            else:
                pct = v * 100.0  # 0〜1 → %
                s = f"{pct:.0f}"
                if bold_best and v == max_val:
                    s = r"\textbf{" + s + "}"
            vals.append(s)
        return vals

    lines = []
    lines.append(r"Setting & Robot & CMA-ES & VISF & DISF (ours) \\")
    lines.append(r"\midrule")

    # -------- Known-shape ブロック --------
    setting = "Known-shape"
    lines.append(r"\multirow{3}{*}{Known-shape}")
    for robot in robot_order:
        row_vals = pivot.loc[(setting, robot)]
        cma_str, visf_str, disf_str = format_row(row_vals, bold_best=True)
        # multirow なので Setting カラムは空セル
        lines.append(
            f" & {robot} & {cma_str} & {visf_str} & {disf_str} \\\\"
        )

    lines.append(r"\cmidrule(lr){2-5}")
    avg_vals = avg_over_robots.loc[setting]
    cma_str, visf_str, disf_str = format_row(avg_vals, bold_best=True)
    lines.append(
        r" & \textbf{Average over robots}"
        + f" & {cma_str} & {visf_str} & {disf_str} \\\\"
    )

    # -------- Observed-shape ブロック --------
    lines.append(r"\midrule")
    setting = "Observed-shape"
    lines.append(r"\multirow{3}{*}{Observed-shape}")
    for robot in robot_order:
        row_vals = pivot.loc[(setting, robot)]
        cma_str, visf_str, disf_str = format_row(row_vals, bold_best=True)
        lines.append(
            f" & {robot} & {cma_str} & {visf_str} & {disf_str} \\\\"
        )

    lines.append(r"\cmidrule(lr){2-5}")
    avg_vals = avg_over_robots.loc[setting]
    cma_str, visf_str, disf_str = format_row(avg_vals, bold_best=True)
    lines.append(
        r" & \textbf{Average over robots}"
        + f" & {cma_str} & {visf_str} & {disf_str} \\\\"
    )

    # -------- Total Average 行 --------
    lines.append(r"\midrule")
    cma_str, visf_str, disf_str = format_row(total_avg, bold_best=True)
    lines.append(
        r"\multicolumn{2}{l}{\textbf{Total Average}}"
        + f" & {cma_str} & {visf_str} & {disf_str} \\\\"
    )

    # 出力
    print("% ---- Robot platform table body ----")
    for line in lines:
        print(line)


if __name__ == "__main__":
    build_robot_platform_latex(CSV_PATH)
