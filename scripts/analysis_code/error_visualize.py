import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. モックアップデータの定義
#    ※ここを実データに置き換えて使う想定
# ------------------------------------------------------------

# オブジェクト名（論文用に短めのラベルにしてます）
objects_known = ["T-shape", "RubberDuck", "Hammer", "WineGlass", "OldCamera"]
objects_obs   = [
    "006_mustard", "011_banana", "029_plate", "033_spatula", "035_drill",
    "037_scissors", "042_wrench", "052_clamp", "058_golf", "065_j_cups",
]

methods = ["CMA-ES", "VISF", "DISF"]

# 形状誤差 E_geom（小さいほど良い）: すべて「3ロボット平均済み」という想定
# shape: (n_methods, n_objects)
known_geom = np.array([
    [0.020, 0.018, 0.022, 0.019, 0.021],  # CMA-ES
    [0.025, 0.023, 0.026, 0.024, 0.025],  # VISF
    [0.018, 0.017, 0.019, 0.017, 0.018],  # DISF
])

obs_geom = np.array([
    [0.050, 0.045, 0.060, 0.055, 0.052, 0.048, 0.058, 0.053, 0.050, 0.057],
    [0.070, 0.065, 0.075, 0.080, 0.072, 0.068, 0.078, 0.074, 0.071, 0.079],
    [0.040, 0.038, 0.045, 0.047, 0.042, 0.039, 0.044, 0.043, 0.041, 0.046],
])

# CoM 誤差 E_CoM（小さいほど良い）
known_com = np.array([
    [0.015, 0.016, 0.017, 0.018, 0.016],  # CMA-ES
    [0.020, 0.021, 0.022, 0.021, 0.020],  # VISF
    [0.008, 0.009, 0.010, 0.009, 0.008],  # DISF
])

obs_com = np.array([
    [0.030, 0.032, 0.035, 0.034, 0.033, 0.031, 0.036, 0.034, 0.032, 0.035],
    [0.045, 0.048, 0.050, 0.052, 0.049, 0.047, 0.051, 0.050, 0.048, 0.052],
    [0.015, 0.016, 0.018, 0.019, 0.017, 0.016, 0.018, 0.017, 0.016, 0.018],
])

# ------------------------------------------------------------
# 2. 描画設定
# ------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharey=False)
# fig, axes = plt.subplots(2, 2, figsize=(11, 6), constrained_layout=True)

bar_width = 0.25
method_positions = np.arange(len(methods))  # x 軸は method ベースにする

def plot_per_object(ax, errors, objects, title, ylabel):
    """
    各オブジェクトについて、method ごとの棒グラフを横並びで描く。
    errors: shape (n_methods, n_objects)
    """
    n_methods, n_objects = errors.shape
    x = np.arange(n_objects)

    for m_idx, method in enumerate(methods):
        # 棒を左右にずらして3本並べる
        offset = (m_idx - (n_methods - 1) / 2) * bar_width
        ax.bar(
            x + offset,
            errors[m_idx],
            width=bar_width,
            label=method if ax is axes[0, 0] else None,  # 凡例は最初のサブ図だけ
        )

    ax.set_xticks(x)
    ax.set_xticklabels(objects, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_yscale("log")  # 誤差がオーダー違いなら log スケールが見やすい

# Known-shape: E_geom, E_CoM
plot_per_object(
    axes[0, 0],
    known_geom,
    objects_known,
    title="Known-shape (3D CAD) – $E_{\\mathrm{geom}}$",
    ylabel="$E_{\\mathrm{geom}}$",
)
plot_per_object(
    axes[0, 1],
    known_com,
    objects_known,
    title="Known-shape (3D CAD) – $E_{\\mathrm{CoM}}$",
    ylabel="$E_{\\mathrm{CoM}}$",
)

# Observed-shape: E_geom, E_CoM
plot_per_object(
    axes[1, 0],
    obs_geom,
    objects_obs,
    title="Observed-shape (YCB) – $E_{\\mathrm{geom}}$",
    ylabel="$E_{\\mathrm{geom}}$",
)
plot_per_object(
    axes[1, 1],
    obs_com,
    objects_obs,
    title="Observed-shape (YCB) – $E_{\\mathrm{CoM}}$",
    ylabel="$E_{\\mathrm{CoM}}$",
)

# 凡例（図全体で1つ）
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=3,
    bbox_to_anchor=(0.5, 1.02),
    # bbox_to_anchor=(0.5, 1.0),
)

# plt.tight_layout()

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
# plt.savefig("error_analysis_mock.pdf", bbox_inches="tight")
