#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import OUTPUTS_ROOT, safe_filename  # noqa: E402


plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Liberation Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot AMPLoc PPO decision summaries from storyline and sample JSON files.")
    parser.add_argument("--storylines-json", type=Path, required=True, help="Path to the class-level storyline summary JSON file.")
    parser.add_argument("--star-samples-json", type=Path, required=True, help="Path to the sample-level PPO trace JSON file.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_ROOT / "amploc_analysis" / "ppo_decision", help="Directory where the figure and table will be written.")
    return parser


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def build_summary_dataframe(storylines: list[dict[str, object]], star_samples: dict[str, list[dict[str, object]]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    storyline_rows = []
    sample_rows = []

    for entry in storylines:
        class_name = str(entry.get("class", "Unknown"))
        preferences = entry.get("preferences", {})
        if not isinstance(preferences, dict):
            preferences = {}

        class_samples = star_samples.get(class_name, []) if isinstance(star_samples, dict) else []
        dominant_action = None
        dominant_action_frequency = None
        if class_samples:
            action_vectors = [tuple(sample.get("Action_Vector", [])) for sample in class_samples if isinstance(sample.get("Action_Vector"), list)]
            if action_vectors:
                action_counter = Counter(action_vectors)
                dominant_action, dominant_count = action_counter.most_common(1)[0]
                dominant_action_frequency = dominant_count / len(class_samples)

        storyline_rows.append(
            {
                "class": class_name,
                "preferred_channels": " | ".join(map(str, entry.get("preferred_channels", []))),
                "preferred_frequency": float(entry.get("preferred_frequency", 0.0)),
                "avoided_channels": " | ".join(map(str, entry.get("avoided_channels", []))),
                "avoided_frequency": float(entry.get("avoided_frequency", 0.0)),
                "sample_count": len(class_samples),
                "dominant_action_vector": str(dominant_action) if dominant_action is not None else None,
                "dominant_action_frequency": dominant_action_frequency,
                "mean_log_probability": float(np.mean([sample.get("Log_Probability", np.nan) for sample in class_samples])) if class_samples else None,
                "mean_state_value": float(np.mean([sample.get("State_Value", np.nan) for sample in class_samples])) if class_samples else None,
                "mean_sequence_length": float(np.mean([sample.get("Sequence_Length", np.nan) for sample in class_samples])) if class_samples else None,
            }
        )

        for channel_name, value in preferences.items():
            sample_rows.append(
                {
                    "class": class_name,
                    "channel_combination": str(channel_name),
                    "preference": float(value),
                }
            )

    storyline_df = pd.DataFrame(storyline_rows)
    preference_df = pd.DataFrame(sample_rows)
    return storyline_df, preference_df


def plot_decision_analysis(storyline_df: pd.DataFrame, preference_df: pd.DataFrame, output_path: Path) -> None:
    if storyline_df.empty or preference_df.empty:
        raise ValueError("The PPO decision inputs are empty.")

    class_order = storyline_df.sort_values("preferred_frequency", ascending=False)["class"].tolist()
    combo_order = preference_df.groupby("channel_combination")["preference"].mean().sort_values(ascending=False).index.tolist()

    preference_matrix = (
        preference_df.pivot_table(index="class", columns="channel_combination", values="preference", aggfunc="mean")
        .reindex(index=class_order, columns=combo_order)
    )

    figure_height = max(7.5, 0.55 * len(class_order) + 3.0)
    fig, axes = plt.subplots(2, 1, figsize=(18, figure_height), gridspec_kw={"height_ratios": [1.5, 1.0]})

    heatmap_ax = axes[0]
    sns.heatmap(
        preference_matrix,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Preference"},
        ax=heatmap_ax,
        annot_kws={"size": 9},
    )
    heatmap_ax.set_title("AMPLoc PPO Decision Heatmap", fontsize=17, fontweight="bold", pad=16)
    heatmap_ax.set_xlabel("Channel combination")
    heatmap_ax.set_ylabel("Class")
    heatmap_ax.tick_params(axis="x", rotation=35)

    bar_ax = axes[1]
    ordered_storylines = storyline_df.set_index("class").reindex(class_order)
    y_positions = np.arange(len(class_order))
    bar_height = 0.34

    preferred_values = ordered_storylines["preferred_frequency"].fillna(0.0).to_numpy(dtype=float)
    dominant_values = ordered_storylines["dominant_action_frequency"].fillna(0.0).to_numpy(dtype=float)

    bars_preferred = bar_ax.barh(
        y_positions - bar_height / 2,
        preferred_values,
        height=bar_height,
        color="#0ea5e9",
        edgecolor="black",
        label="Storyline preferred frequency",
    )
    bars_dominant = bar_ax.barh(
        y_positions + bar_height / 2,
        dominant_values,
        height=bar_height,
        color="#f59e0b",
        edgecolor="black",
        label="Dominant sampled action frequency",
    )

    bar_ax.set_yticks(y_positions)
    bar_ax.set_yticklabels(class_order)
    bar_ax.set_xlim(0, 1.0)
    bar_ax.set_xlabel("Frequency")
    bar_ax.set_title("AMPLoc Storyline vs Sample-Level Dominant Action", fontsize=15, fontweight="bold", pad=12)
    bar_ax.grid(axis="x", alpha=0.25)
    bar_ax.legend(loc="lower right")

    for idx, class_name in enumerate(class_order):
        sample_count = int(ordered_storylines.loc[class_name, "sample_count"])
        preferred_frequency = float(ordered_storylines.loc[class_name, "preferred_frequency"])
        dominant_frequency = float(ordered_storylines.loc[class_name, "dominant_action_frequency"] or 0.0)
        bar_ax.text(
            1.01,
            idx,
            f"n={sample_count}, pref={preferred_frequency:.2f}, dom={dominant_frequency:.2f}",
            va="center",
            ha="left",
            fontsize=9,
            transform=bar_ax.get_yaxis_transform(),
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    storylines = load_json(args.storylines_json)
    star_samples = load_json(args.star_samples_json)
    if not isinstance(storylines, list) or not isinstance(star_samples, dict):
        raise ValueError("Unexpected PPO decision JSON structure.")

    storyline_df, preference_df = build_summary_dataframe(storylines, star_samples)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_slug = safe_filename(args.storylines_json.parent.name)
    table_path = output_dir / f"AMPLoc_ppo_decision_summary_{dataset_slug}.csv"
    plot_path = output_dir / f"AMPLoc_ppo_decision_heatmap_{dataset_slug}.png"

    storyline_df.to_csv(table_path, index=False)
    plot_decision_analysis(storyline_df, preference_df, plot_path)

    print(f"[AMPLoc] Saved PPO summary to {table_path}")
    print(f"[AMPLoc] Saved PPO decision plot to {plot_path}")


if __name__ == "__main__":
    main()