from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt


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


def collect_mass_pattern_files(
    results_dir: str | Path,
    total_mass : float,
) -> list[Path]:
    results_dir = Path(results_dir)
    pattern = f"mass_{total_mass:.2f}__*.npy"
    return sorted(results_dir.glob(pattern))


def extract_plot_data(
    save_paths: list[Path],
) -> tuple[np.ndarray, np.ndarray, list[str], float]:
    x_values    = []
    y_values    = []
    conditions  = []
    total_masses = []

    for save_path in save_paths:
        saved_data = load_saved_result(
            save_path = save_path,
        )

        x_values.append(float(saved_data["signed_normalized_moment_arm_x"]))
        y_values.append(float(saved_data["success_rate"]))
        conditions.append(str(saved_data["condition"]))
        total_masses.append(float(saved_data["total_mass"]))

    x_values = np.asarray(x_values, dtype = float)
    y_values = np.asarray(y_values, dtype = float)

    if len(total_masses) == 0:
        raise ValueError("No result files were loaded.")

    unique_masses = sorted(set(total_masses))
    if len(unique_masses) != 1:
        raise ValueError(
            f"Expected one unique total mass, got {unique_masses}"
        )

    order = np.argsort(x_values)

    x_values   = x_values[order]
    y_values   = y_values[order]
    conditions = [conditions[i] for i in order]

    return x_values, y_values, conditions, unique_masses[0]


def plot_one_mass_sweep(
    results_dir  : str | Path,
    total_mass   : float,
    output_path  : str | Path | None = None,
    show         : bool              = True,
):
    save_paths = collect_mass_pattern_files(
        results_dir = results_dir,
        total_mass  = total_mass,
    )
    if len(save_paths) == 0:
        raise FileNotFoundError(
            f"No files found for total mass {total_mass:.2f} in {results_dir}"
        )

    x_values, y_values, conditions, loaded_total_mass = extract_plot_data(
        save_paths = save_paths,
    )

    plt.figure(figsize = (7.0, 4.8))
    plt.plot(
        x_values,
        y_values,
        marker = "o",
        label  = f"total mass = {loaded_total_mass:.2f}",
    )

    for x, y, condition in zip(x_values, y_values, conditions):
        plt.annotate(
            condition,
            (x, y),
            textcoords = "offset points",
            xytext     = (0, 8),
            ha         = "center",
            fontsize   = 9,
        )

    plt.xlabel("Signed normalized moment arm")
    plt.ylabel("Success rate")
    plt.ylim(-0.05, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents = True, exist_ok = True)
        plt.savefig(output_path, dpi = 300)

    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    # results_dir = "./results/pose_perturbation_mass_sweep"
    results_dir = "/home/cudagl/dataset/RAS_results/box5/"
    total_mass  = 0.50

    plot_one_mass_sweep(
        results_dir = results_dir,
        total_mass  = total_mass,
        output_path = "/home/cudagl/dataset/RAS_results/box5/success_rate_vs_signed_normalized_moment_arm_mass_0.50.png",
        show        = True,
    )
