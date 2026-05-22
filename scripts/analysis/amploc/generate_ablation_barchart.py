#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import OUTPUTS_ROOT, safe_filename  # noqa: E402


plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Liberation Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False
plt.style.use("seaborn-v0_8-whitegrid")


FULL_MODEL_SCORES = {
    "Dataset 1": 0.8106,
    "Dataset 2": 0.7951,
}

ABLATION_SCORES = {
    "w/o RL Adaptation": {"Dataset 1": 0.7538, "Dataset 2": 0.7335},
    "w/o Agent (Static Fusion)": {"Dataset 1": 0.7311, "Dataset 2": 0.7088},
    "w/o RPI Modality": {"Dataset 1": 0.7152, "Dataset 2": 0.6953},
    "w/o Structure Modality": {"Dataset 1": 0.7795, "Dataset 2": 0.7633},
    "w/o Sequence Modality": {"Dataset 1": 0.6903, "Dataset 2": 0.6719},
    "Sequence Modality Only": {"Dataset 1": 0.7405, "Dataset 2": 0.7196},
    "w/o H-CAFN (use MLP Fusion)": {"Dataset 1": 0.7654, "Dataset 2": 0.7482},
    "w/ Simple Reward": {"Dataset 1": 0.7961, "Dataset 2": 0.7804},
}


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an AMPLoc ablation bar chart from the benchmark Ave-F1 values.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_ROOT / "amploc_analysis" / "ablation", help="Directory where the figure and table will be written.")
    parser.add_argument("--save-pdf", action="store_true", help="Also save a PDF version of the chart.")
    return parser


def build_ablation_table() -> pd.DataFrame:
    rows = []
    for variant, scores in ABLATION_SCORES.items():
        row = {"variant": variant}
        for dataset_name, score in scores.items():
            full_score = FULL_MODEL_SCORES[dataset_name]
            row[f"{dataset_name}_AveF1"] = score
            row[f"{dataset_name}_drop_points"] = (full_score - score) * 100.0
        rows.append(row)

    return pd.DataFrame(rows)


def plot_ablation_chart(table: pd.DataFrame, output_path: Path, save_pdf: bool) -> None:
    variants = table["variant"].tolist()
    dataset1_drop = table["Dataset 1_drop_points"].to_numpy(dtype=float)
    dataset2_drop = table["Dataset 2_drop_points"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(15, 8))
    x = np.arange(len(variants))
    bar_width = 0.36

    bars1 = ax.bar(x - bar_width / 2, dataset1_drop, bar_width, label="Dataset 1", color="#38bdf8", edgecolor="black")
    bars2 = ax.bar(x + bar_width / 2, dataset2_drop, bar_width, label="Dataset 2", color="#fb923c", edgecolor="black")

    ax.set_title("AMPLoc Component Ablation Study", fontsize=18, fontweight="bold", pad=18)
    ax.set_ylabel("Ave-F1 Drop (Percentage Points)", fontsize=13)
    ax.set_xlabel("Ablated Component", fontsize=13, labelpad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=40, ha="right", fontsize=11)
    ax.set_ylim(0, max(dataset1_drop.max(), dataset2_drop.max()) * 1.18)
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    ax.bar_label(bars1, padding=3, fmt="%.2f", fontsize=9)
    ax.bar_label(bars2, padding=3, fmt="%.2f", fontsize=9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    if save_pdf:
        pdf_path = output_path.with_suffix(".pdf")
        plt.savefig(pdf_path, dpi=300, bbox_inches="tight", format="pdf")
        print(f"[AMPLoc] Saved ablation chart to {pdf_path}")
    plt.close(fig)


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    table = build_ablation_table()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "AMPLoc_ablation_study_table.csv"
    png_path = output_dir / "AMPLoc_ablation_study_barchart.png"
    table.to_csv(csv_path, index=False)

    plot_ablation_chart(table, png_path, args.save_pdf)

    print(f"[AMPLoc] Saved ablation table to {csv_path}")
    print(f"[AMPLoc] Saved ablation chart to {png_path}")


if __name__ == "__main__":
    main()