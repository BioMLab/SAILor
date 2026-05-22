#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    OUTPUTS_ROOT,
    discover_run_directories,
    ensure_parent_dir,
    load_run_history,
    run_label,
    safe_filename,
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract AMPLoc training history from run logs.")
    parser.add_argument("--outputs-dir", type=Path, default=OUTPUTS_ROOT, help="Root directory that contains run logs.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory where CSV summaries will be written.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Process a single run directory.")
    parser.add_argument("--log-path", type=Path, default=None, help="Process a single log file.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on the number of run directories processed.")
    return parser


def resolve_run_directories(args: argparse.Namespace) -> list[Path]:
    if args.log_path is not None:
        return [args.log_path.parent]
    if args.run_dir is not None:
        return [args.run_dir]

    run_dirs = discover_run_directories(args.outputs_dir)
    if args.limit is not None:
        return run_dirs[: args.limit]
    return run_dirs


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    output_dir = args.output_dir or (args.outputs_dir / "amploc_analysis" / "training_history")
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = resolve_run_directories(args)
    summary_rows: list[dict[str, object]] = []

    for run_dir in run_dirs:
        log_path = args.log_path if args.log_path is not None else run_dir / "run.log"
        if not log_path.exists():
            print(f"[AMPLoc] Skipping missing log: {log_path}")
            continue

        df, summary = load_run_history(run_dir)
        if df.empty:
            print(f"[AMPLoc] No epoch metrics found in: {log_path}")
            continue

        label = run_label(run_dir, args.outputs_dir)
        csv_path = output_dir / f"AMPLoc_training_history_{safe_filename(label)}.csv"
        ensure_parent_dir(csv_path)
        df.to_csv(csv_path, index=False)

        summary_rows.append(
            {
                "run_name": label,
                "log_path": str(log_path),
                **summary,
            }
        )
        print(f"[AMPLoc] Saved training history to {csv_path}")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_csv = output_dir / "AMPLoc_training_history_summary.csv"
        summary_df.to_csv(summary_csv, index=False)
        print(f"[AMPLoc] Saved summary to {summary_csv}")
    else:
        print("[AMPLoc] No usable run logs were found.")


if __name__ == "__main__":
    main()