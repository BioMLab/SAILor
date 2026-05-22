#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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


METRICS = ["Ave-F1", "MiP", "MiR", "MiF", "MaAUC"]


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot AMPLoc evaluation summaries from JSON files.")
    parser.add_argument("--summary-json", type=Path, required=True, help="Path to a JSON summary file.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_ROOT / "amploc_analysis" / "metric_summaries", help="Directory where the figure and table will be written.")
    return parser


def load_summary(summary_json: Path) -> dict[str, object]:
    return json.loads(summary_json.read_text(encoding="utf-8"))


def extract_metric_block(summary: dict[str, object], mean_key: str, std_key: str) -> tuple[dict[str, float], dict[str, float]] | None:
    mean_block = summary.get(mean_key)
    std_block = summary.get(std_key)
    if not isinstance(mean_block, dict) or not isinstance(std_block, dict):
        return None

    means = {metric: float(mean_block[metric]) for metric in METRICS if metric in mean_block}
    stds = {metric: float(std_block[metric]) for metric in METRICS if metric in std_block}
    if not means:
        return None
    return means, stds


def extract_cv_summary(summary: dict[str, object]) -> tuple[str, list[tuple[str, dict[str, float], dict[str, float]]]]:
    dataset_label = str(summary.get("dataset", summary.get("dataset_name", "summary")))

    if "average_metrics" in summary and "std_metrics" in summary:
        block = extract_metric_block(summary, "average_metrics", "std_metrics")
        if block is not None:
            means, stds = block
            return dataset_label, [("5-fold Internal CV", means, stds)]

    if "all_fold_metrics" in summary and isinstance(summary["all_fold_metrics"], list) and summary["all_fold_metrics"]:
        fold_frame = pd.DataFrame(summary["all_fold_metrics"])
        means = {metric: float(fold_frame[metric].mean()) for metric in METRICS if metric in fold_frame.columns}
        stds = {metric: float(fold_frame[metric].std(ddof=0)) for metric in METRICS if metric in fold_frame.columns}
        return dataset_label, [("5-fold Internal CV", means, stds)]

    raise ValueError("The JSON summary does not look like a cross-validation result file.")


def extract_rigorous_summary(summary: dict[str, object]) -> tuple[str, list[tuple[str, dict[str, float], dict[str, float]]]]:
    dataset_label = str(summary.get("dataset", summary.get("dataset_name", "summary")))
    panels = []

    for panel_name, mean_key, std_key in [
        ("Validation", "val_mean", "val_std"),
        ("Test", "test_mean", "test_std"),
    ]:
        block = extract_metric_block(summary, mean_key, std_key)
        if block is not None:
            means, stds = block
            panels.append((panel_name, means, stds))

    if not panels:
        raise ValueError("The JSON summary does not look like a rigorous evaluation result file.")

    return dataset_label, panels


def plot_panels(dataset_label: str, panels: list[tuple[str, dict[str, float], dict[str, float]]], output_path: Path) -> None:
    fig, axes = plt.subplots(1, len(panels), figsize=(7.5 * len(panels), 6.0), squeeze=False)
    axes_list = axes[0].tolist()

    max_value = 0.0
    for panel_name, means, _ in panels:
        for metric in METRICS:
            max_value = max(max_value, means.get(metric, 0.0))

    for ax, (panel_name, means, stds) in zip(axes_list, panels):
        values = np.array([means.get(metric, np.nan) for metric in METRICS], dtype=float)
        errors = np.array([stds.get(metric, np.nan) for metric in METRICS], dtype=float)
        x_positions = np.arange(len(METRICS))

        bars = ax.bar(
            x_positions,
            values,
            yerr=errors,
            capsize=5,
            color="#3b82f6",
            edgecolor="#1e3a8a",
            linewidth=0.8,
            alpha=0.92,
        )
        ax.set_xticks(x_positions)
        ax.set_xticklabels(METRICS, rotation=25, ha="right")
        ax.set_ylim(0, max(1.0, max_value * 1.15))
        ax.set_ylabel("Score")
        ax.set_title(f"AMPLoc {dataset_label} - {panel_name}", fontsize=14, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        for bar, value in zip(bars, values):
            if not np.isfinite(value):
                continue
            ax.text(bar.get_x() + bar.get_width() / 2, value + max_value * 0.015, f"{value:.4f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_table(panels: list[tuple[str, dict[str, float], dict[str, float]]], output_path: Path) -> None:
    rows = []
    for panel_name, means, stds in panels:
        for metric in METRICS:
            if metric not in means:
                continue
            rows.append(
                {
                    "panel": panel_name,
                    "metric": metric,
                    "mean": means.get(metric),
                    "std": stds.get(metric),
                }
            )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    summary = load_summary(args.summary_json)

    try:
        dataset_label, panels = extract_cv_summary(summary)
    except ValueError:
        dataset_label, panels = extract_rigorous_summary(summary)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    slug = safe_filename(dataset_label)
    table_path = output_dir / f"AMPLoc_metric_summary_{slug}.csv"
    plot_path = output_dir / f"AMPLoc_metric_summary_{slug}.png"

    write_table(panels, table_path)
    plot_panels(dataset_label, panels, plot_path)

    print(f"[AMPLoc] Saved metric table to {table_path}")
    print(f"[AMPLoc] Saved metric plot to {plot_path}")


if __name__ == "__main__":
    main()