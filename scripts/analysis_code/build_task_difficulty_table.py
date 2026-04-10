import pandas as pd

CSV_PATH = "/home/cudagl/data/RAS_results/panda_summary_enriched.csv"  # 適宜パスを書き換えてください


def build_task_difficulty_latex(csv_path: str) -> None:
    df = pd.read_csv(csv_path)

    # --- どのオブジェクトがどの Setting か ---
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

    # LaTeX 上での表示名（必要に応じて編集）
    display_name_map = {
        "custom_T": "T-shape Block",
        "custom_RubberDuck": "Rubber Duck",
        "custom_Hammer": "Hammer",
        "custom_WineGlass": "Wine Glass",
        "custom_OldCamera": "Old Camera",
        "006_mustard_bottle": r"006\_mustard\_bottle",
        "011_banana": r"011\_banana",
        "029_plate": r"029\_plate",
        "033_spatula": r"033\_spatula",
        "035_power_drill": r"035\_power\_drill",
        "037_scissors": r"037\_scissors",
        "042_adjustable_wrench": r"042\_adjustable\_wrench",
        "052_extra_large_clamp": r"052\_extra\_large\_clamp",
        "058_golf_ball": r"058\_golf\_ball",
        "065-j_cups": r"065\text{-}j\_cups",
    }

    def disp(name: str) -> str:
        """オブジェクト名を \texttt{} でくるんだ LaTeX 表示に変換"""
        base = display_name_map.get(name, name.replace("_", r"\_"))
        return r"\texttt{" + base + "}"

    def to_check(val: str) -> str:
        """CSV 中の記号を LaTeX 用に変換（✓ → $\greencheck$）"""
        v = str(val).strip()
        if v == "✓":
            return r"$\greencheck$"
        return "-"

    # DataFrame を Setting ごとにインデックス化
    df_known = df[df["object"].isin(known_objects)].set_index("object")
    df_obs   = df[df["object"].isin(observed_objects)].set_index("object")

    lines = []
    lines.append(r"Setting & Object & CMA-ES & VISF & DISF (ours) \\")
    lines.append(r"\midrule")

    # ---------------- Known-shape ブロック ----------------
    lines.append(r"\multirow{5}{*}{Known-shape}")
    for obj in known_objects:
        row = df_known.loc[obj]
        line = (
            f" & {disp(obj)}"
            f" & {to_check(row['cma_check'])}"
            f" & {to_check(row['visf_check'])}"
            f" & {to_check(row['disf_check'])} \\\\"
        )
        lines.append(line)

    # ---------------- Observed-shape ブロック ----------------
    lines.append(r"\midrule")
    lines.append(r"\multirow{10}{*}{Observed-shape}")
    for obj in observed_objects:
        row = df_obs.loc[obj]
        line = (
            f" & {disp(obj)}"
            f" & {to_check(row['cma_check'])}"
            f" & {to_check(row['visf_check'])}"
            f" & {to_check(row['disf_check'])} \\\\"
        )
        lines.append(line)

    # ---------------- サマリ行 ----------------
    lines.append(r"\midrule")

    def object_success_fraction(sub_df: pd.DataFrame, method_prefix: str) -> str:
        """✓ が付いているオブジェクト数 / 全オブジェクト数 を 'a/b' で返す"""
        col = f"{method_prefix}_check"
        n_success = (sub_df[col].map(lambda v: str(v).strip()) == "✓").sum()
        total = len(sub_df)
        return f"{n_success}/{total}"

    # Success rate (Known-shape)
    known_rates = [
        object_success_fraction(df_known, "cma"),
        object_success_fraction(df_known, "visf"),
        object_success_fraction(df_known, "disf"),
    ]
    lines.append(
        r"\multicolumn{2}{l}{Success rate (Known-shape)}"
        + " & " + " & ".join(known_rates) + r" \\"
    )

    # Success rate (Observed-shape)
    obs_rates = [
        object_success_fraction(df_obs, "cma"),
        object_success_fraction(df_obs, "visf"),
        object_success_fraction(df_obs, "disf"),
    ]
    lines.append(
        r"\multicolumn{2}{l}{Success rate (Observed-shape)}"
        + " & " + " & ".join(obs_rates) + r" \\"
    )

    # Planning time [ms]（Panda 全オブジェクト平均）
    planning_times = []
    for prefix in ["cma", "visf", "disf"]:
        mean_t = df[f"{prefix}_avg_time_ms"].mean()
        planning_times.append(f"{mean_t:.1f}")
    lines.append(
        r"\multicolumn{2}{l}{Planning time [ms]}"
        + " & " + " & ".join(planning_times) + r" \\"
    )

    # ------------- 出力 -------------
    print("% ---- Task difficulty table body (for Panda) ----")
    for line in lines:
        print(line)

if __name__ == "__main__":
    build_task_difficulty_latex(CSV_PATH)
