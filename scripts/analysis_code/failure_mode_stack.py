"""
Observed-shape only: failure-mode breakdown (horizontal stacked bars)
- y-axis: methods
- x-axis: counts
- stacks: failure modes (+ optional Success)

Matplotlib only, no seaborn.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# 1) Configuration
# -----------------------------
OBSERVED_SETTING_NAME = "Observed-shape"   # label shown in title

# METHODS: List[str] = ["CMA-ES", "VISF", "DISF (ours)"]
METHODS: List[str] = [ "DISF (ours)", "VISF", "CMA-ES"]

# Stacked components (failure modes)
MODES: List[str] = [
    "Contact-induced abort",
    # "Reachability rejection",
    "Misgrasp",   # or "Misgrasp" if you prefer
]

INCLUDE_SUCCESS: bool = True
SUCCESS_LABEL: str = "Success"

# Manual colors (optional but recommended)
MODE_COLORS: Dict[str, str] = {
    "Contact-induced abort": "#d11818",
    # "Reachability rejection": "#f1bd10",
    "Misgrasp": "#e5ae22",
    "Success": "#2ca02c",
}


# -------------------------------------
# 2) Fill aggregated counts (Observed)
# -------------------------------------
# COUNTS[method][mode] = count
# If INCLUDE_SUCCESS is True, include COUNTS[method]["Success"] too.
#
# Replace placeholder values with your real log aggregation.
COUNTS: Dict[str, Dict[str, int]] = {
    "CMA-ES": {
        "Contact-induced abort": 5,
        # "Reachability rejection": 0,
        "Misgrasp": 4,
        "Success": 0,
    },
    "VISF": {
        "Contact-induced abort": 4,
        # "Reachability rejection": 2,
        "Misgrasp": 3,
        "Success": 2,
    },
    "DISF (ours)": {
        "Contact-induced abort": 0,
        # "Reachability rejection": 1,
        "Misgrasp": 1,
        "Success": 8,
    },
}


# -----------------------------
# 3) Plotting implementation
# -----------------------------
@dataclass
class PlotStyle:
    figsize: Tuple[float, float] = (15.8, 3.)
    ylabel: str = ""
    xlabel: str = "Count"
    title: str = "Failure-mode breakdown (Observed-shape, real execution)"
    legend_ncol: int = 3
    dpi: int = 300
    bar_height: float = 0.55


def _validate_counts(
    counts: Dict[str, Dict[str, int]],
    methods: List[str],
    modes: List[str],
    include_success: bool,
    success_label: str,
) -> List[str]:
    required = list(modes) + ([success_label] if include_success else [])
    missing = []
    for m in methods:
        if m not in counts:
            missing.append(f"Missing method: {m}")
            continue
        for k in required:
            if k not in counts[m]:
                missing.append(f"Missing key '{k}' for method: {m}")
    return missing


def plot_observed_failure_modes_barh(
    counts: Dict[str, Dict[str, int]],
    methods: List[str],
    modes: List[str],
    include_success: bool = True,
    success_label: str = "Success",
    mode_colors: Optional[Dict[str, str]] = None,
    style: PlotStyle = PlotStyle(),
    save_path: str | None = "failure_mode_breakdown_observed.pdf",
) -> None:
    missing = _validate_counts(counts, methods, modes, include_success, success_label)
    if missing:
        raise ValueError("Invalid COUNTS:\n- " + "\n- ".join(missing))

    stack_labels = list(modes) + ([success_label] if include_success else [])

    # Color mapping (manual required keys; fallback to Matplotlib default cycle)
    if mode_colors is not None:
        missing_c = [k for k in stack_labels if k not in mode_colors]
        if missing_c:
            raise ValueError(
                "mode_colors is missing keys for: " + ", ".join(missing_c) +
                "\nExpected keys: " + ", ".join(stack_labels)
            )
        color_of = lambda label: mode_colors[label]
    else:
        default_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        if len(default_cycle) < len(stack_labels):
            default_cycle = (default_cycle * (len(stack_labels) // max(len(default_cycle), 1) + 1))[: len(stack_labels)]
        auto_map = {label: default_cycle[i] for i, label in enumerate(stack_labels)}
        color_of = lambda label: auto_map[label]

    # Prepare data in method order
    y = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=style.figsize)

    left = np.zeros(len(methods), dtype=float)
    for label in stack_labels:
        vals = np.array([counts[m][label] for m in methods], dtype=float)
        ax.barh(
            y,
            vals,
            left=left,
            height=style.bar_height,
            color=color_of(label),
            label=label,
        )
        left += vals

    ax.set_yticks(y)
    ax.set_yticklabels(methods, fontsize=20)
    ax.set_xlabel(style.xlabel, fontsize=20)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=18)
    if style.ylabel:
        ax.set_ylabel(style.ylabel, fontsize=20)
    # ax.set_title(style.title, y=1.02)

    # Make grid paper-friendly
    ax.grid(axis="x", linestyle="-", linewidth=0.5, alpha=0.6)

    # # Optional: show totals at bar ends
    # totals = left
    # for i, total in enumerate(totals):
    #     ax.text(total + 0.05, i, f"{int(total)}", va="center", ha="left", fontsize=9)



    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),   # center-top above axes
        ncol=3,                       # 2列くらいが見やすい（必要なら4でもOK）
        frameon=True,
        fontsize=18,
    )
    # fig.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))  # leave space on the right for legend
    # fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    fig.subplots_adjust(top=0.86)

    if save_path is not None:
        fig.savefig(save_path, dpi=style.dpi, bbox_inches="tight")
        print(f"[Saved] {save_path}")

    plt.show()


if __name__ == "__main__":
    plot_observed_failure_modes_barh(
        counts=COUNTS,
        methods=METHODS,
        modes=MODES,
        include_success=INCLUDE_SUCCESS,
        success_label=SUCCESS_LABEL,
        mode_colors=MODE_COLORS,
        save_path="./failure_mode_breakdown_observed.pdf",
    )
