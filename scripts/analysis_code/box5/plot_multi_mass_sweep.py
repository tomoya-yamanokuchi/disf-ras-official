from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt



from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 3) Plotting implementation
# -----------------------------
@dataclass
class PlotStyle:
    figsize    : Tuple[float, float] = (15.8, 3.)
    ylabel     : str   = "Grasp success rate"
    # xlabel     : str   = "Normalized moment"
    xlabel: str = "Normalized grasp-relative CoM displacement"
    title      : str   = ""
    legend_ncol: int   = 3
    dpi        : int   = 300
    bar_height : float = 0.55



def load_saved_result(
    save_path: str | Path,
) -> dict:
    save_path = Path(save_path)
    loaded = np.load(
        save_path,
        allow_pickle = True,
    )

    if not isinstance(loaded, np.ndarray) or loaded.shape != ():
        raise ValueError(
            f"Expected a scalar object array from np.save(dict), got shape={getattr(loaded, 'shape', None)}"
        )

    saved_data = loaded.item()
    if not isinstance(saved_data, dict):
        raise ValueError(f"Expected dict, got {type(saved_data)}")

    return saved_data


def infer_total_mass_from_filename(
    save_path: str | Path,
) -> float:
    save_path = Path(save_path)
    match = re.search(r"mass_([0-9]+(?:\.[0-9]+)?)__", save_path.stem)
    if match is None:
        raise ValueError(f"Could not infer total mass from filename: {save_path.name}")
    return float(match.group(1))


def collect_result_files(
    results_dir: str | Path,
) -> list[Path]:
    results_dir = Path(results_dir)
    return sorted(results_dir.glob("mass_*.npy"))


def group_files_by_total_mass(
    save_paths: list[Path],
) -> dict[float, list[Path]]:
    grouped: dict[float, list[Path]] = {}

    for save_path in save_paths:
        total_mass = infer_total_mass_from_filename(
            save_path = save_path,
        )
        if total_mass not in grouped:
            grouped[total_mass] = []
        grouped[total_mass].append(save_path)

    return grouped


def extract_curve_from_save_paths(
    save_paths: list[Path],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    x_values   = []
    y_values   = []
    conditions = []

    for save_path in save_paths:
        saved_data = load_saved_result(
            save_path = save_path,
        )

        x_values.append(float(saved_data["signed_normalized_moment_arm_x"]))
        y_values.append(float(saved_data["success_rate"]))
        conditions.append(str(saved_data["condition"]))

    x_values = np.asarray(x_values, dtype = float)
    y_values = np.asarray(y_values, dtype = float)

    order = np.argsort(x_values)

    x_values   = x_values[order]
    y_values   = y_values[order]
    conditions = [conditions[i] for i in order]

    return x_values, y_values, conditions


def plot_multi_mass_sweep(
    results_dir        : str | Path,
    target_total_masses: list[float] | None = None,
    output_path        : str | Path | None  = None,
    show               : bool               = True,
    annotate_points    : bool               = False,
):
    save_paths = collect_result_files(
        results_dir = results_dir,
    )
    if len(save_paths) == 0:
        raise FileNotFoundError(f"No result files were found in {results_dir}")

    grouped = group_files_by_total_mass(
        save_paths = save_paths,
    )

    if target_total_masses is None:
        target_total_masses = sorted(grouped.keys())
    else:
        target_total_masses = sorted(target_total_masses)

    # plt.figure(figsize = (7.5, 5.0))
    # fig, ax = plt.subplots(figsize=(7.5, 4.0))
    fig, ax = plt.subplots(figsize=(6, 4.0))

    for total_mass in target_total_masses:
        if total_mass not in grouped:
            print(f"[Warning] total mass {total_mass:.2f} was not found. Skipped.")
            continue

        x_values, y_values, conditions = extract_curve_from_save_paths(
            save_paths = grouped[total_mass],
        )

        ax.plot(
            x_values,
            y_values,
            marker     = "o",
            markersize = 8,
            linewidth  = 1.5,
            alpha      = 0.6,
            label      = f"mass = {total_mass:.1f}",
        )

        if annotate_points:
            for x, y, condition in zip(x_values, y_values, conditions):
                ax.annotate(
                    condition,
                    (x, y),
                    textcoords = "offset points",
                    xytext     = (0, 8),
                    ha         = "center",
                    fontsize   = 8,
                )

    # plt.xlabel("Signed normalized moment arm")

    style = PlotStyle()

    ax.set_xlabel(style.ylabel)
    ax.set_ylabel(style.xlabel)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True)

    ax.grid(linestyle="--", linewidth=0.5, alpha=0.6)

    ax.legend(fontsize=13, loc="lower right", bbox_to_anchor=(1.0, 0.43,))

    ax.set_ylabel(style.ylabel, fontsize=16)
    ax.set_xlabel(style.xlabel, fontsize=16)

    ax.set_yticklabels(ax.get_yticklabels(), fontsize=15)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=15)
    # ax.set_tight_layout()
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents = True, exist_ok = True)
        fig.savefig(output_path, dpi = 300)

    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    plot_multi_mass_sweep(
        # results_dir         = "./results/pose_perturbation_mass_sweep",
        results_dir         = "/home/cudagl/dataset/RAS_results/box5_original/",
        # target_total_masses = [0.30, 0.50, 0.70],
        target_total_masses = [0.4, 0.6, 0.8, 1.0, 1.2],
        output_path         = "/home/cudagl/dataset/RAS_results/box5_original/success_rate_vs_signed_normalized_moment_arm_multi_mass.pdf",
        show                = False,
        annotate_points     = False,
    )
