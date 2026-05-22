#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import OUTPUTS_ROOT, safe_filename, select_metric_column  # noqa: E402


plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Liberation Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot AMPLoc training curves from extracted CSV history files.")
    parser.add_argument("--history-dir", type=Path, default=OUTPUTS_ROOT / "amploc_analysis" / "training_history", help="Directory that contains extracted CSV history files.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory where plots will be written.")
    parser.add_argument("--csv", type=Path, action="append", default=None, help="Optional explicit CSV file paths. Can be passed multiple times.")
    return parser


def load_history_frames(history_dir: Path, csv_paths: list[Path] | None) -> list[tuple[str, pd.DataFrame]]:
    if csv_paths:
        selected_paths = csv_paths
    else:
        selected_paths = sorted(history_dir.glob("AMPLoc_training_history_*.csv"))

    frames: list[tuple[str, pd.DataFrame]] = []
    for csv_path in selected_paths:
        if csv_path.name == "AMPLoc_training_history_summary.csv" or not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if not df.empty:
            frames.append((csv_path.stem.replace("AMPLoc_training_history_", ""), df))
    return frames


def plot_single_run(df: pd.DataFrame, run_name: str, output_path: Path) -> None:
    epochs = pd.to_numeric(df["epoch"], errors="coerce")
    train_metric_name = select_metric_column(df, ["train_ave_f1", "train_f1"])
    val_metric_name = select_metric_column(df, ["val_ave_f1", "val_f1"])

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    ax1 = axes[0, 0]
    if "train_loss" in df.columns and df["train_loss"].notna().any():
        ax1.plot(epochs, df["train_loss"], label="Train Loss", linewidth=2, marker="o")
    if "val_loss" in df.columns and df["val_loss"].notna().any():
        ax1.plot(epochs, df["val_loss"], label="Validation Loss", linewidth=2, marker="s")
    ax1.set_title(f"AMPLoc Training Loss - {run_name}", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    if train_metric_name is not None:
        ax2.plot(epochs, df[train_metric_name], label=train_metric_name.replace("_", " ").title(), linewidth=2, marker="o")
    if val_metric_name is not None:
        ax2.plot(epochs, df[val_metric_name], label=val_metric_name.replace("_", " ").title(), linewidth=2, marker="s")
    ax2.set_title(f"AMPLoc F1 Curve - {run_name}", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])

    ax3 = axes[1, 0]
    if val_metric_name is not None:
        val_series = pd.to_numeric(df[val_metric_name], errors="coerce")
        ax3.plot(epochs, val_series, label=f"Validation {val_metric_name}", linewidth=2, marker="o", color="green")
        best_idx = val_series.idxmax()
        best_epoch = int(df.loc[best_idx, "epoch"])
        best_value = float(val_series.loc[best_idx])
        ax3.plot(best_epoch, best_value, "ro", markersize=12, label=f"Best: {best_value:.4f} @ Epoch {best_epoch}")
        ax3.axvline(x=best_epoch, color="r", linestyle="--", alpha=0.4)
        ax3.axhline(y=best_value, color="r", linestyle="--", alpha=0.4)
    ax3.set_title(f"AMPLoc Validation Performance - {run_name}", fontsize=14, fontweight="bold")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Score")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])

    ax4 = axes[1, 1]
    if "val_loss" in df.columns and df["val_loss"].notna().any() and val_metric_name is not None:
        ax4_twin = ax4.twinx()
        ax4.plot(epochs, df["val_loss"], color="tab:blue", linewidth=2, marker="o", label="Validation Loss")
        ax4_twin.plot(epochs, pd.to_numeric(df[val_metric_name], errors="coerce"), color="tab:red", linewidth=2, marker="s", label="Validation Score")
        ax4.set_ylabel("Validation Loss", color="tab:blue")
        ax4_twin.set_ylabel("Validation Score", color="tab:red")
        ax4.tick_params(axis="y", labelcolor="tab:blue")
        ax4_twin.tick_params(axis="y", labelcolor="tab:red")
        ax4.set_title(f"AMPLoc Loss vs Performance - {run_name}", fontsize=14, fontweight="bold")
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(frames: list[tuple[str, pd.DataFrame]], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=False)

    ax1, ax2 = axes
    for run_name, df in frames:
        epochs = pd.to_numeric(df["epoch"], errors="coerce")
        val_metric_name = select_metric_column(df, ["val_ave_f1", "val_f1"])
        if "val_loss" in df.columns and df["val_loss"].notna().any():
            ax1.plot(epochs, df["val_loss"], linewidth=2, label=run_name)
        if val_metric_name is not None:
            ax2.plot(epochs, pd.to_numeric(df[val_metric_name], errors="coerce"), linewidth=2, label=run_name)

    ax1.set_title("AMPLoc Validation Loss Comparison", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation Loss")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.set_title("AMPLoc Validation Score Comparison", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Score")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    output_dir = args.output_dir or (args.history_dir.parent / "plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = load_history_frames(args.history_dir, args.csv)
    if not frames:
        print("[AMPLoc] No training history CSV files were found.")
        return

    for run_name, df in frames:
        run_output = output_dir / f"AMPLoc_training_curves_{safe_filename(run_name)}.png"
        plot_single_run(df, run_name, run_output)
        print(f"[AMPLoc] Saved plot to {run_output}")

    if len(frames) > 1:
        comparison_output = output_dir / "AMPLoc_training_curves_comparison.png"
        plot_comparison(frames, comparison_output)
        print(f"[AMPLoc] Saved comparison plot to {comparison_output}")


if __name__ == "__main__":
    main()