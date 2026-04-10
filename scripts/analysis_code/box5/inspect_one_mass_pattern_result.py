from __future__ import annotations

from pathlib import Path
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


def print_saved_result(
    saved_data: dict,
) -> None:
    print("=" * 80)
    print("Loaded result")
    print("=" * 80)

    for key, value in saved_data.items():
        print(f"{key:30s}: {value}")


def plot_one_result(
    saved_data  : dict,
    output_path : str | Path | None = None,
    show        : bool              = True,
) -> None:
    x = float(saved_data["signed_normalized_moment_arm_x"])
    y = float(saved_data["success_rate"])

    label = (
        f"condition={saved_data['condition']}, "
        f"mass={saved_data['total_mass']:.2f}"
    )

    plt.figure(figsize = (6.0, 4.5))
    plt.scatter(
        [x],
        [y],
        s     = 80,
        label = label,
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
    save_path = "/home/cudagl/dataset/RAS_results/box5/mass_0.50__uniform.npy"

    saved_data = load_saved_result(
        save_path = save_path,
    )
    print_saved_result(
        saved_data = saved_data,
    )
    plot_one_result(
        saved_data  = saved_data,
        output_path = "/home/cudagl/dataset/RAS_results/box5/mass_0.50__uniform.png",
        show        = True,
    )
