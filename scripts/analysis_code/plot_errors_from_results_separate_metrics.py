#!/usr/bin/env python3
"""Save *separate* figures for E_geom and E_CoM.

Each metric gets its own figure with a 2x1 layout:
  Top:    Known-shape
  Bottom: Observed-shape

This is intended for papers where the combined 2x2 figure becomes too small.

Search pattern:
  <results_dir>/<robot>/<method>/<object>/E_Geom.npy
  <results_dir>/<robot>/<method>/<object>/E_CoM.npy

Aggregates multiple files by taking the mean per (robot, method, object).

Usage:
  python scripts/analysis/plot_errors_from_results_separate_metrics.py \
    --results-dir /home/cudagl/data/RAS_results --robot panda \
    --out-geom-png /home/cudagl/data/RAS_results/Egeom_by_setting.png \
    --out-com-png  /home/cudagl/data/RAS_results/ECoM_by_setting.png \
    --out-geom-pdf /home/cudagl/data/RAS_results/Egeom_by_setting.pdf \
    --out-com-pdf  /home/cudagl/data/RAS_results/ECoM_by_setting.pdf

Notes:
- Colors: keep the E_CoM palette as-is; update PALETTE_E_GEOM to your preferred set.
- If you want a strict paper width, pass --fig-width and --fig-height.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, LogFormatterMathtext


KNOWN = ["custom_T", "custom_RubberDuck", "custom_Hammer", "custom_WineGlass", "custom_OldCamera"]
OBSERVED = [
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

METHODS = ["cma", "visf", "disf"]
METHOD_LABELS = ["CMA-ES", "VISF", "DISF"]

# --- palettes ---
# Swap E_geom palette to something more "modern" if you like.
PALETTE_E_COM = ["#1D3557", "#D4A017", "#B23A48"]  # navy / amber / brick
PALETTE_E_GEOM  = ["#7E57C2", "#EC407A", "#26A69A"]  # purple / pink / teal


# -------------------------- style utils --------------------------
def apply_modern_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "DejaVu Sans",
        "font.size": 20, # 11,
        "axes.titlesize": 20, # 12.5,
        "axes.labelsize": 20 , #11,
        "xtick.labelsize": 20, # 9.5,
        "ytick.labelsize": 20, # 10,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "axes.axisbelow": True,
        "legend.frameon": False,
        "legend.fontsize": 20, # 10,
        "xtick.major.size": 3.5,
        "xtick.major.width": 0.8,
        "ytick.major.size": 3.5,
        "ytick.major.width": 0.8,
    })


def _format_known_label(obj: str) -> str:
    name = obj.replace("custom_", "")
    mapping = {
        "T": "T-shape Block",
        "RubberDuck": "Rubber Duck",
        "WineGlass": "Wine Glass",
        "OldCamera": "Old Camera",
    }
    return mapping.get(name, name.replace("_", " "))


def _format_observed_label(obj: str) -> str:
    s = obj.replace("-", "_")
    parts = s.split("_")
    if len(parts) >= 2 and parts[0].isdigit():
        idx = parts[0]
        name = " ".join(parts[1:])
        return f"{idx}\n{name}"
    return obj.replace("_", " ")


def _set_log_scale_if_needed(ax, errors: np.ndarray) -> None:
    vals = errors[np.isfinite(errors)]
    vals = vals[vals > 0]
    if vals.size == 0:
        return

    maxv = float(np.max(vals))
    minv = float(np.min(vals))

    if maxv / max(1e-30, minv) >= 1e2:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10.0))
        ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
        ax.grid(True, which="major", axis="y")
        ax.grid(True, which="minor", axis="y", alpha=0.12)

        lo = 10 ** (math.floor(math.log10(minv)) - 0.25)
        hi = 10 ** (math.ceil(math.log10(maxv)) + 0.15)
        ax.set_ylim(lo, hi)
    else:
        ax.grid(True, axis="y")
        ax.set_ylim(0.0, maxv * 1.15)


def _method_patches(palette: list[str]) -> list[Patch]:
    return [Patch(facecolor=palette[i], edgecolor="none", label=METHOD_LABELS[i]) for i in range(len(METHOD_LABELS))]


# -------------------------- data collection --------------------------
def collect_errors(results_dir: Path, robot: str) -> pd.DataFrame:
    results_dir = results_dir.expanduser().resolve()
    data = []

    for method in METHODS:
        for obj in KNOWN + OBSERVED:
            base = results_dir / robot / method / obj
            geom_vals: list[float] = []
            com_vals: list[float] = []

            if base.exists():
                for p in base.rglob("E_Geom.npy"):
                    try:
                        arr = np.asarray(np.load(p, allow_pickle=True)).ravel()
                        if arr.size > 0:
                            geom_vals.extend([float(x) for x in arr.tolist()])
                    except Exception:
                        continue

                for p in base.rglob("E_CoM.npy"):
                    try:
                        arr = np.asarray(np.load(p, allow_pickle=True)).ravel()
                        if arr.size > 0:
                            com_vals.extend([float(x) for x in arr.tolist()])
                    except Exception:
                        continue

            geom_mean = float(np.mean(geom_vals)) if len(geom_vals) > 0 else float("nan")
            com_mean = float(np.mean(com_vals)) if len(com_vals) > 0 else float("nan")

            data.append({
                "robot": robot,
                "method": method,
                "object": obj,
                "E_geom": geom_mean,
                "E_CoM": com_mean,
            })

    return pd.DataFrame.from_records(data)


def make_error_arrays(df: pd.DataFrame):
    n_methods = len(METHODS)

    known_geom = np.full((n_methods, len(KNOWN)), np.nan)
    known_com  = np.full((n_methods, len(KNOWN)), np.nan)
    obs_geom   = np.full((n_methods, len(OBSERVED)), np.nan)
    obs_com    = np.full((n_methods, len(OBSERVED)), np.nan)

    for i, m in enumerate(METHODS):
        for j, obj in enumerate(KNOWN):
            row = df[(df["method"] == m) & (df["object"] == obj)]
            if not row.empty:
                known_geom[i, j] = row["E_geom"].values[0]
                known_com[i, j]  = row["E_CoM"].values[0]

        for j, obj in enumerate(OBSERVED):
            row = df[(df["method"] == m) & (df["object"] == obj)]
            if not row.empty:
                obs_geom[i, j] = row["E_geom"].values[0]
                obs_com[i, j]  = row["E_CoM"].values[0]

    return known_geom, known_com, obs_geom, obs_com


# -------------------------- plotting --------------------------
def _plot_metric_figure(
    known: np.ndarray,
    observed: np.ndarray,
    metric_tex: str,
    palette: list[str],
    out_png: Path,
    out_pdf: Path | None,
    fig_width: float,
    fig_height: float,
) -> None:
    apply_modern_style()

    known_labels = [_format_known_label(o) for o in KNOWN]
    obs_labels   = [_format_observed_label(o) for o in OBSERVED]

    fig, axes = plt.subplots(
        2, 1,
        figsize=(fig_width, fig_height),
        constrained_layout=True,
        sharey=False,
    )

    bar_width = 0.23

    def plot_grouped(ax, errors: np.ndarray, labels: list[str], title: str, rotate: int, ha: str):
        n_methods, n_objects = errors.shape
        x = np.arange(n_objects)

        for m_idx in range(n_methods):
            offset = (m_idx - (n_methods - 1) / 2.0) * bar_width
            ax.bar(
                x + offset,
                errors[m_idx],
                width=bar_width,
                color=palette[m_idx],
                alpha=0.95,
                linewidth=0.0,
                zorder=3,
            )

        # ax.set_title(title, pad=4)
        ax.set_title(title, loc="center", pad=6)

        ax.set_ylabel(metric_tex)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=rotate, ha=ha)

        ax.margins(x=0.01)
        _set_log_scale_if_needed(ax, errors)
        ax.grid(False, axis="x")
        ax.grid(True, axis="y")

    plot_grouped(axes[0], known, known_labels, f"Known-shape", rotate=0,  ha="center")
    plot_grouped(axes[1], observed, obs_labels, f"Observed-shape", rotate=45, ha="right")

    # One legend per figure (top center)
    axes[0].legend(
        handles=_method_patches(palette),
        loc="lower center",
        bbox_to_anchor=(0.5, 1.4),
        ncol=3,
        title=f"{metric_tex} colors",
        frameon=True,
        borderaxespad=0.0,
        handlelength=1.6,
        columnspacing=1.4,
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    print(f"Saved: {out_png}")

    if out_pdf is not None:
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_pdf, bbox_inches="tight")
        print(f"Saved: {out_pdf}")

    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, default=Path("/home/cudagl/data/RAS_results"))
    p.add_argument("--robot", type=str, default="panda")

    p.add_argument("--out-geom-png", type=Path, default=Path("/home/cudagl/data/RAS_results/errors_Egeom.png"))
    p.add_argument("--out-com-png",  type=Path, default=Path("/home/cudagl/data/RAS_results/errors_ECoM.png"))
    p.add_argument("--out-geom-pdf", type=Path, default=None)
    p.add_argument("--out-com-pdf",  type=Path, default=None)

    # Paper sizing knobs
    p.add_argument("--fig-width", type=float, default=13.0)
    p.add_argument("--fig-height", type=float, default=8)

    args = p.parse_args()

    df = collect_errors(args.results_dir, args.robot)
    known_geom, known_com, obs_geom, obs_com = make_error_arrays(df)

    _plot_metric_figure(
        known=known_geom,
        observed=obs_geom,
        metric_tex="$E_{\\mathrm{geom}}$",
        palette=PALETTE_E_GEOM,
        out_png=args.out_geom_png,
        out_pdf=args.out_geom_pdf,
        fig_width=args.fig_width,
        fig_height=args.fig_height,
    )

    _plot_metric_figure(
        known=known_com,
        observed=obs_com,
        metric_tex="$E_{\\mathrm{CoM}}$",
        palette=PALETTE_E_COM,
        out_png=args.out_com_png,
        out_pdf=args.out_com_pdf,
        fig_width=args.fig_width,
        fig_height=args.fig_height,
    )


if __name__ == "__main__":
    main()
