#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import OUTPUTS_ROOT, safe_filename  # noqa: E402


plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Liberation Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot AMPLoc RBP importance from a summary JSON file.")
    parser.add_argument("--summary-json", type=Path, required=True, help="Path to an RBP importance summary JSON file.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_ROOT / "amploc_analysis" / "rbp_importance", help="Directory where the plot and table will be written.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of top RBPs to plot.")
    return parser


def load_summary(summary_json: Path) -> dict[str, object]:
    content = summary_json.read_text(encoding="utf-8")
    return json.loads(content)


def build_dataframe(summary: dict[str, object], top_k: int) -> pd.DataFrame:
    entries = summary.get("top_10_rbps", [])
    if not isinstance(entries, list):
        return pd.DataFrame(columns=["RBP", "Importance_Mean", "Importance_Std", "Importance_Max", "Importance_Min"])

    df = pd.DataFrame(entries)
    if df.empty:
        return df

    if "Importance_Mean" in df.columns:
        df = df.sort_values("Importance_Mean", ascending=False)

    df = df.head(top_k).reset_index(drop=True)
    return df


def plot_rbp_importance(df: pd.DataFrame, dataset_label: str, num_samples: int | None, num_rbps: int | None, output_path: Path) -> None:
    if df.empty:
        raise ValueError("No RBP importance rows available for plotting.")

    plot_height = max(6.0, 0.55 * len(df) + 2.0)
    fig, ax = plt.subplots(figsize=(12.5, plot_height))

    y_positions = np.arange(len(df))
    means = df["Importance_Mean"].to_numpy(dtype=float)
    stds = df["Importance_Std"].to_numpy(dtype=float) if "Importance_Std" in df.columns else None

    ax.barh(
        y_positions,
        means,
        xerr=stds,
        color="#2b6cb0",
        alpha=0.92,
        capsize=4,
        edgecolor="#1a365d",
        linewidth=0.8,
    )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(df["RBP"].astype(str))
    ax.invert_yaxis()
    ax.set_xlabel("Importance Mean", fontsize=12)
    ax.set_title(f"AMPLoc RBP Importance - {dataset_label}", fontsize=16, fontweight="bold", pad=16)
    ax.grid(axis="x", alpha=0.25)

    max_mean = float(np.nanmax(means)) if len(means) else 0.0
    label_offset = max(0.0005, max_mean * 0.02)
    for index, value in enumerate(means):
        ax.text(value + label_offset, index, f"{value:.4f}", va="center", ha="left", fontsize=9)

    subtitle_parts = []
    if num_samples is not None:
        subtitle_parts.append(f"Samples: {num_samples}")
    if num_rbps is not None:
        subtitle_parts.append(f"RBPs: {num_rbps}")
    if subtitle_parts:
        ax.text(0.0, -0.10, " | ".join(subtitle_parts), transform=ax.transAxes, fontsize=10, va="top")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    summary = load_summary(args.summary_json)
    dataset_label = str(summary.get("dataset", args.summary_json.stem))
    num_samples = summary.get("num_samples")
    num_rbps = summary.get("num_rbps")

    df = build_dataframe(summary, args.top_k)
    if df.empty:
        print("[AMPLoc] No RBP importance entries were found in the JSON summary.")
        return

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_slug = safe_filename(dataset_label)
    table_path = output_dir / f"AMPLoc_rbp_importance_{dataset_slug}_top{args.top_k}.csv"
    plot_path = output_dir / f"AMPLoc_rbp_importance_{dataset_slug}_top{args.top_k}.png"

    df.to_csv(table_path, index=False)
    plot_rbp_importance(df, dataset_label, num_samples if isinstance(num_samples, int) else None, num_rbps if isinstance(num_rbps, int) else None, plot_path)

    print(f"[AMPLoc] Saved RBP table to {table_path}")
    print(f"[AMPLoc] Saved RBP plot to {plot_path}")


if __name__ == "__main__":
    main()