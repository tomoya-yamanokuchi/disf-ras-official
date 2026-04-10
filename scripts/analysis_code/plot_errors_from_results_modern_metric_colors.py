#!/usr/bin/env python3
"""
Modern 2x2 bar plots for E_Geom and E_CoM, with *metric-specific* color palettes.

- E_geom subplots use one palette
- E_CoM  subplots use another palette

Search pattern:
  <results_dir>/<robot>/<method>/<object>/E_Geom.npy (and E_CoM.npy)
Aggregates multiple files by taking the mean per (robot, method, object).

Usage:
  python scripts/analysis/plot_errors_from_results_modern_metric_colors.py \
    --results-dir /home/cudagl/data/RAS_results --robot panda

Outputs:
  PNG: <results_dir>/errors_by_setting_modern_metric_colors.png (default)
  PDF: optional via --out-pdf
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

METHODS       = ["cma", "visf", "disf"]
METHOD_LABELS = ["CMA-ES", "VISF", "DISF"]

# Two palettes (one per metric). Picked to be paper-friendly and clearly separated.
# Feel free to swap colors; only the order must match METHOD_LABELS.
# PALETTE_E_GEOM = ["#4C78A8", "#F58518", "#54A24B"]  # blue / orange / green
PALETTE_E_GEOM =  ["#7E57C2", "#EC407A", "#26A69A"]  # purple / pink / teal　
# PALETTE_E_GEOM = ["#2F2F2F", "#C9A227", "#C7512C"]
# PALETTE_E_COM = ["#334E68", "#C7A27C", "#7B2C3B"]
PALETTE_E_COM  = ["#1D3557", "#D4A017", "#B23A48"]


# -------------------------- style utils --------------------------
def apply_modern_style() -> None:
    """A clean, paper-friendly style without relying on seaborn."""
    plt.rcParams.update({
        # figure
        "figure.dpi"      : 200,
        "savefig.dpi"     : 300,
        "figure.facecolor": "white",
        "axes.facecolor"  : "white",
        # fonts
        "font.family"    : "DejaVu Sans",
        "font.size"      : 11,
        "axes.titlesize" : 12.5,
        "axes.labelsize" : 11,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 10,
        # axes
        "axes.linewidth"   : 0.8,
        "axes.spines.top"  : False,
        "axes.spines.right": False,
        "axes.grid"        : True,
        "grid.alpha"       : 0.25,
        "grid.linewidth"   : 0.8,
        "grid.linestyle"   : "-",
        "axes.axisbelow"   : True,
        # legend
        "legend.frameon" : True,
        # "legend.frameon" : False,
        "legend.fontsize": 10,
        # ticks
        "xtick.major.size" : 3.5,
        "xtick.major.width": 0.8,
        "ytick.major.size" : 3.5,
        "ytick.major.width": 0.8,
    })


def _format_known_label(obj: str) -> str:
    name = obj.replace("custom_", "")
    mapping = {
        "T"         : "T-shape Block",
        "RubberDuck": "Rubber Duck",
        "WineGlass" : "Wine Glass",
        "OldCamera" : "Old Camera",
    }
    return mapping.get(name, name.replace("_", " "))


def _format_observed_label(obj: str) -> str:
    # e.g., "006_mustard_bottle" -> "006\nmustard bottle"
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

    # log-scale when spanning >= 2 orders of magnitude (robust)
    if maxv / max(1e-30, minv) >= 1e2:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10.0))
        ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
        ax.grid(True, which="major", axis="y")
        ax.grid(True, which="minor", axis="y", alpha=0.12)

        # pad y-limits a bit on log scale
        lo = 10 ** (math.floor(math.log10(minv)) - 0.25)
        hi = 10 ** (math.ceil(math.log10(maxv)) + 0.15)
        ax.set_ylim(lo, hi)
    else:
        ax.grid(True, axis="y")
        ax.set_ylim(0.0, maxv * 1.15)


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
                "n_geom_samples": len(geom_vals),
                "n_com_samples": len(com_vals),
            })

    return pd.DataFrame.from_records(data)


def make_error_arrays(df: pd.DataFrame):
    known_objs = KNOWN
    obs_objs = OBSERVED
    n_methods = len(METHODS)

    known_geom = np.full((n_methods, len(known_objs)), np.nan)
    known_com  = np.full((n_methods, len(known_objs)), np.nan)
    obs_geom   = np.full((n_methods, len(obs_objs)), np.nan)
    obs_com    = np.full((n_methods, len(obs_objs)), np.nan)

    for i, m in enumerate(METHODS):
        for j, obj in enumerate(known_objs):
            row = df[(df["method"] == m) & (df["object"] == obj)]
            if not row.empty:
                known_geom[i, j] = row["E_geom"].values[0]
                known_com[i, j]  = row["E_CoM"].values[0]

        for j, obj in enumerate(obs_objs):
            row = df[(df["method"] == m) & (df["object"] == obj)]
            if not row.empty:
                obs_geom[i, j] = row["E_geom"].values[0]
                obs_com[i, j]  = row["E_CoM"].values[0]

    return known_geom, known_com, obs_geom, obs_com


# -------------------------- plotting --------------------------
def _method_patches(palette: list[str]) -> list[Patch]:
    # return [Patch(facecolor=palette[i], edgecolor="none", label=METHOD_LABELS[i]) for i in range(len(METHOD_LABELS))]
    return [Patch(facecolor=palette[i], edgecolor="none", label=METHOD_LABELS[i]) for i in range(len(METHOD_LABELS))]


def plot_errors(
    known_geom: np.ndarray,
    known_com: np.ndarray,
    obs_geom: np.ndarray,
    obs_com: np.ndarray,
    out_png: Path,
    out_pdf: Path | None = None,
) -> None:
    apply_modern_style()

    objects_known_labels = [_format_known_label(o) for o in KNOWN]
    objects_obs_labels   = [_format_observed_label(o) for o in OBSERVED]

    fig, axes = plt.subplots(
        2, 2,
        figsize=(13.0, 5.2),
        constrained_layout=True,
        sharey=False,
    )

    bar_width = 0.23

    def plot_grouped_bars(
        ax,
        errors: np.ndarray,
        objects: list[str],
        title: str,
        ylabel: str,
        palette: list[str],
    ):
        n_methods, n_objects = errors.shape
        x = np.arange(n_objects)

        for m_idx, method in enumerate(METHOD_LABELS):
            offset = (m_idx - (n_methods - 1) / 2.0) * bar_width
            vals = errors[m_idx]

            ax.bar(
                x + offset,
                vals,
                width=bar_width,
                color=palette[m_idx],
                alpha=0.95,
                linewidth=0.0,
                zorder=3,
            )

        ax.set_title(title, loc="center")
        ax.set_ylabel(ylabel)

        ax.set_xticks(x)

        if "Known" in title:
            ax.set_xticklabels(objects, rotation=0, ha="center")
        else:
            # observed: keep angled labels (readable) but compact
            ax.set_xticklabels(objects, rotation=45, ha="right")

        ax.margins(x=0.01)
        _set_log_scale_if_needed(ax, errors)

        # subtle x-grid off, y-grid on
        ax.grid(False, axis="x")
        ax.grid(True, axis="y")

    # Draw bars with different palettes per metric
    plot_grouped_bars(axes[0, 0], known_geom, objects_known_labels, "Known-shape: $E_{\\mathrm{geom}}$", "$E_{\\mathrm{geom}}$", PALETTE_E_GEOM)
    plot_grouped_bars(axes[0, 1], known_com,  objects_known_labels, "Known-shape: $E_{\\mathrm{CoM}}$",  "$E_{\\mathrm{CoM}}$",  PALETTE_E_COM)
    plot_grouped_bars(axes[1, 0], obs_geom,   objects_obs_labels,   "Observed-shape: $E_{\\mathrm{geom}}$", "$E_{\\mathrm{geom}}$", PALETTE_E_GEOM)
    plot_grouped_bars(axes[1, 1], obs_com,    objects_obs_labels,   "Observed-shape: $E_{\\mathrm{CoM}}$",  "$E_{\\mathrm{CoM}}$",  PALETTE_E_COM)

    # Two small legends: one per column (metric)

    bbox_to_anchor = (0.5, 1.2)

    axes[0, 0].legend(
        handles=_method_patches(PALETTE_E_GEOM),
        loc="lower center",
        bbox_to_anchor=bbox_to_anchor,
        ncol=3,
        title="$E_{\\mathrm{geom}}$ colors",
    )
    axes[0, 1].legend(
        handles=_method_patches(PALETTE_E_COM),
        loc="lower center",
        bbox_to_anchor=bbox_to_anchor,
        ncol=3,
        title="$E_{\\mathrm{CoM}}$ colors",
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
    p.add_argument("--out-png", type=Path, default=Path("/home/cudagl/data/RAS_results/errors_by_setting_modern_metric_colors.png"))
    p.add_argument("--out-pdf", type=Path, default=None)
    args = p.parse_args()

    df = collect_errors(args.results_dir, args.robot)
    known_geom, known_com, obs_geom, obs_com = make_error_arrays(df)
    plot_errors(known_geom, known_com, obs_geom, obs_com, args.out_png, args.out_pdf)


if __name__ == "__main__":
    main()
