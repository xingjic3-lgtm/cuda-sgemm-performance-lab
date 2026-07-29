#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


MATRIX_SIZES = [128, 256, 512, 1024, 2048, 4096]
KERNEL_NAMES = {
    1: "Naive",
    2: "GMEM coalescing",
    3: "SMEM caching",
    4: "1D block tiling",
    5: "2D block tiling",
    6: "Vectorized access",
    7: "Bank layout",
    8: "Padded layout",
    9: "Static tuning",
    10: "Warp tiling (best config/size)",
    11: "Split double buffer",
    12: "Async double buffer",
}


def load_baseline(csv_path: Path) -> dict[int, dict[int, float]]:
    results = {kernel_id: {} for kernel_id in KERNEL_NAMES}
    with csv_path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            kernel_id = int(row["kernel_id"])
            size = int(row["size_m"])
            if (
                kernel_id in results
                and row["status"] == "ok"
                and size in MATRIX_SIZES
                and int(row["size_n"]) == size
                and int(row["size_k"]) == size
            ):
                results[kernel_id][size] = float(row["gflops"]) / 1000.0
    return results


def load_best_k10(
    csv_path: Path,
) -> tuple[dict[int, float], dict[int, int]]:
    candidates: dict[int, list[tuple[float, int]]] = {
        size: [] for size in MATRIX_SIZES
    }
    with csv_path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            size = int(row["size_m"])
            if (
                row["record_type"] == "phase3_tuning"
                and row["status"] == "ok"
                and size in candidates
                and int(row["size_n"]) == size
                and int(row["size_k"]) == size
            ):
                candidates[size].append(
                    (float(row["kernel_gflops"]) / 1000.0, int(row["config_index"]))
                )

    best_tflops: dict[int, float] = {}
    best_configs: dict[int, int] = {}
    for size, rows in candidates.items():
        if not rows:
            raise ValueError(f"No successful Phase 3 K10 result for {size}^3")
        best_tflops[size], best_configs[size] = max(rows)
    return best_tflops, best_configs


def validate(results: dict[int, dict[int, float]]) -> None:
    required = set(MATRIX_SIZES)
    for kernel_id, by_size in results.items():
        missing = sorted(required - set(by_size))
        if missing:
            raise ValueError(f"Kernel {kernel_id} is missing sizes: {missing}")


def plot(
    results: dict[int, dict[int, float]],
    best_configs: dict[int, int],
    output_path: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#64748b",
            "axes.labelcolor": "#1e293b",
            "xtick.color": "#334155",
            "ytick.color": "#334155",
        }
    )

    fig, ax = plt.subplots(figsize=(15, 8), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x_positions = list(range(len(MATRIX_SIZES)))
    colors = list(plt.get_cmap("tab20").colors[:12])
    markers = ["o", "s", "^", "v", "P", "X", "D", "<", ">", "*", "h", "p"]

    for kernel_id in KERNEL_NAMES:
        values = [results[kernel_id][size] for size in MATRIX_SIZES]
        is_k10 = kernel_id == 10
        is_experiment = kernel_id in (11, 12)
        ax.plot(
            x_positions,
            values,
            label=f"K{kernel_id}: {KERNEL_NAMES[kernel_id]}",
            color="#e65100" if is_k10 else colors[kernel_id - 1],
            linewidth=3.4 if is_k10 else 1.8,
            linestyle="--" if is_experiment else "-",
            marker=markers[kernel_id - 1],
            markersize=8 if is_k10 else 5.5,
            markeredgecolor="white",
            markeredgewidth=0.8,
            alpha=1.0 if is_k10 else 0.86,
            zorder=5 if is_k10 else 2,
        )

    for x, size in enumerate(MATRIX_SIZES):
        value = results[10][size]
        ax.annotate(
            f"{value:.2f}\nC{best_configs[size]}",
            (x, value),
            xytext=(0, 11),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color="#9a3412",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xticks(x_positions, [str(size) for size in MATRIX_SIZES])
    ax.set_xlim(-0.12, len(MATRIX_SIZES) - 0.88)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Square matrix size (M = N = K)")
    ax.set_ylabel("Throughput (TFLOP/s)")
    ax.set_title("FP32 GEMM Performance by Kernel and Matrix Size — RTX 5060 Ti")
    ax.grid(axis="y", color="#cbd5e1", linewidth=0.8, alpha=0.75)
    ax.grid(axis="x", color="#e2e8f0", linewidth=0.7, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=3,
        frameon=False,
        fontsize=9.5,
        handlelength=2.5,
        columnspacing=1.5,
    )

    fig.text(
        0.5,
        0.012,
        "K10 uses the best successful Phase 3 configuration at each size; "
        "all other lines use reports/baseline_5060ti.csv.",
        ha="center",
        color="#475569",
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 1))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-input",
        type=Path,
        default=project_root / "reports" / "baseline_5060ti.csv",
    )
    parser.add_argument(
        "--tuning-input",
        type=Path,
        default=project_root / "reports" / "phase3_tuning_results.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "benchmark_results.png",
    )
    args = parser.parse_args()

    results = load_baseline(args.baseline_input)
    results[10], best_configs = load_best_k10(args.tuning_input)
    validate(results)
    plot(results, best_configs, args.output)

    selected = ", ".join(
        f"{size}:C{best_configs[size]}" for size in MATRIX_SIZES
    )
    print(f"Wrote {args.output}")
    print(f"K10 selections: {selected}")


if __name__ == "__main__":
    main()
