#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import OUTPUTS_ROOT, safe_filename  # noqa: E402


plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Liberation Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")


HEATMAP_DATA = {
    "a": {
        "dataset_label": "Dataset A",
        "title": "AMPLoc Benchmark Heatmap - Dataset A",
        "output_slug": "dataset_a",
        "figsize": (10, 8),
        "rows": [
            ["AMPLoc", 0.92, 0.88, 0.78, 0.66],
            ["LncTracker", 0.91, 0.87, 0.72, 0.60],
            ["LncMamba", 0.88, 0.85, 0.65, 0.56],
            ["LncLocFormer", 0.86, 0.83, 0.63, 0.54],
            ["CFPLncLoc", 0.75, 0.71, 0.68, 0.64],
            ["GraphLncLoc", 0.75, 0.70, 0.50, 0.40],
            ["DeepLncLoc", 0.50, 0.45, 0.30, 0.25],
            ["LncLocator 1.0", 0.45, 0.40, 0.22, 0.18],
        ],
        "columns": ["Nucleus", "Cytoplasm", "Chromatin", "Insoluble Cytoplasm"],
    },
    "b": {
        "dataset_label": "Dataset B",
        "title": "AMPLoc Benchmark Heatmap - Dataset B",
        "output_slug": "dataset_b",
        "figsize": (14, 8),
        "rows": [
            ["AMPLoc", 0.91, 0.90, 0.82, 0.80, 0.75, 0.72, 0.86],
            ["LncTracker", 0.93, 0.92, 0.72, 0.70, 0.68, 0.65, 0.85],
            ["LncMamba", 0.90, 0.88, 0.65, 0.63, 0.60, 0.58, 0.80],
            ["LncLocFormer", 0.88, 0.86, 0.68, 0.66, 0.62, 0.60, 0.82],
            ["CFPLncLoc", 0.70, 0.68, 0.65, 0.63, 0.71, 0.69, 0.68],
            ["GraphLncLoc", 0.75, 0.72, 0.55, 0.52, 0.50, 0.48, 0.70],
            ["DeepLncLoc", 0.45, 0.42, 0.32, 0.30, 0.28, 0.26, 0.40],
            ["LncLocator 1.0", 0.40, 0.38, 0.28, 0.25, 0.23, 0.21, 0.35],
        ],
        "columns": ["Nucleus", "Cytoplasm", "Nucleoplasm", "Cytosol", "Nucleolus", "Chromatin", "Membrane"],
    },
}


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot AMPLoc benchmark heatmaps for the reference datasets.")
    parser.add_argument("--dataset", choices=sorted(HEATMAP_DATA.keys()), required=True, help="Select which benchmark heatmap to render.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_ROOT / "amploc_analysis" / "heatmaps", help="Directory where the heatmap will be written.")
    parser.add_argument("--save-pdf", action="store_true", help="Also save a PDF version alongside the PNG file.")
    return parser


def render_heatmap(dataset_key: str, output_dir: Path, save_pdf: bool) -> None:
    config = HEATMAP_DATA[dataset_key]
    df = pd.DataFrame(config["rows"], columns=["Model", *config["columns"]]).set_index("Model")

    fig, ax = plt.subplots(figsize=config["figsize"])
    sns.heatmap(
        df,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "F1 Score"},
        ax=ax,
        annot_kws={"size": 11},
    )
    ax.set_title(config["title"], fontsize=16, fontweight="bold", pad=18)
    ax.set_xlabel("Compartment", fontsize=12)
    ax.set_ylabel("Model", fontsize=12)
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"AMPLoc_heatmap_{config['output_slug']}.png"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"[AMPLoc] Saved heatmap to {png_path}")

    if save_pdf:
        pdf_path = output_dir / f"AMPLoc_heatmap_{config['output_slug']}.pdf"
        plt.savefig(pdf_path, dpi=300, bbox_inches="tight", format="pdf")
        print(f"[AMPLoc] Saved heatmap to {pdf_path}")

    plt.close(fig)


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    render_heatmap(args.dataset, args.output_dir, args.save_pdf)


if __name__ == "__main__":
    main()