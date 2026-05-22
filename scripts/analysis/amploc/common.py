from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"

FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
EPOCH_PATTERN = re.compile(r"--- Epoch\s+(?P<epoch>\d+)/(?:\d+)\s+---")

PRIMARY_METRIC_PATTERN = re.compile(
    rf"Train(?:ing)? Loss:\s*(?P<train_loss>{FLOAT_PATTERN})\s*,\s*"
    rf"Train(?:ing)? (?:Ave-)?F1:\s*(?P<train_ave_f1>{FLOAT_PATTERN})\s*\|\s*"
    rf"Val(?:idation)? Loss:\s*(?P<val_loss>{FLOAT_PATTERN})\s*,\s*"
    rf"Val(?:idation)? (?:Ave-)?F1:\s*(?P<val_ave_f1>{FLOAT_PATTERN})"
)

SECONDARY_METRIC_PATTERN = re.compile(
    rf"Train(?:ing)? Loss:\s*(?P<train_loss>{FLOAT_PATTERN})\s*,\s*"
    rf"Train(?:ing)? F1:\s*(?P<train_f1>{FLOAT_PATTERN})\s*\|\s*"
    rf"Val(?:idation)? Loss:\s*(?P<val_loss>{FLOAT_PATTERN})\s*,\s*"
    rf"Val(?:idation)? F1:\s*(?P<val_f1>{FLOAT_PATTERN})"
)


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def safe_filename(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")
    return slug or "run"


def run_label(run_dir: Path, outputs_root: Path = OUTPUTS_ROOT) -> str:
    try:
        relative = run_dir.relative_to(outputs_root)
        return "__".join(relative.parts)
    except ValueError:
        return run_dir.name


def discover_run_directories(outputs_root: Path = OUTPUTS_ROOT) -> list[Path]:
    if not outputs_root.exists():
        return []

    run_dirs = []
    for log_path in outputs_root.rglob("run.log"):
        run_dir = log_path.parent
        if run_dir not in run_dirs:
            run_dirs.append(run_dir)

    def sort_key(path: Path) -> str:
        try:
            return str(path.relative_to(outputs_root))
        except ValueError:
            return str(path)

    run_dirs.sort(key=sort_key)
    return run_dirs


def _extract_metrics_from_line(line: str) -> dict[str, float] | None:
    primary = PRIMARY_METRIC_PATTERN.search(line)
    if primary:
        return {key: float(value) for key, value in primary.groupdict().items()}

    secondary = SECONDARY_METRIC_PATTERN.search(line)
    if secondary:
        return {key: float(value) for key, value in secondary.groupdict().items()}

    return None


def parse_training_log(log_path: Path) -> pd.DataFrame:
    try:
        content = log_path.read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        return pd.DataFrame()

    records: list[dict[str, float | int | None]] = []
    current_epoch: Optional[int] = None

    for line in content.splitlines():
        epoch_match = EPOCH_PATTERN.search(line)
        if epoch_match:
            current_epoch = int(epoch_match.group("epoch"))
            continue

        if current_epoch is None:
            continue

        metrics = _extract_metrics_from_line(line)
        if metrics is None:
            continue

        record: dict[str, float | int | None] = {"epoch": current_epoch}
        record.update(metrics)
        records.append(record)
        current_epoch = None

    if not records:
        return pd.DataFrame(
            columns=["epoch", "train_loss", "train_ave_f1", "train_f1", "val_loss", "val_ave_f1", "val_f1"]
        )

    df = pd.DataFrame(records)
    for column in ["train_loss", "train_ave_f1", "train_f1", "val_loss", "val_ave_f1", "val_f1"]:
        if column not in df.columns:
            df[column] = pd.NA

    df = df[["epoch", "train_loss", "train_ave_f1", "train_f1", "val_loss", "val_ave_f1", "val_f1"]]
    return df.sort_values("epoch").reset_index(drop=True)


def select_metric_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for column in candidates:
        if column in df.columns and df[column].notna().any():
            return column
    return None


def summarize_history(df: pd.DataFrame, metric_columns: Iterable[str] = ("val_ave_f1", "val_f1")) -> dict[str, object]:
    if df.empty:
        return {
            "num_epochs": 0,
            "best_metric_name": None,
            "best_metric_value": None,
            "best_epoch": None,
            "final_epoch": None,
            "final_train_loss": None,
            "final_val_loss": None,
            "final_train_ave_f1": None,
            "final_val_ave_f1": None,
            "final_train_f1": None,
            "final_val_f1": None,
        }

    metric_name = select_metric_column(df, metric_columns)
    best_metric_value = None
    best_epoch = None

    if metric_name is not None:
        valid_series = pd.to_numeric(df[metric_name], errors="coerce")
        valid_series = valid_series[valid_series.notna()]
        if not valid_series.empty:
            best_index = valid_series.idxmax()
            best_metric_value = float(valid_series.loc[best_index])
            best_epoch = int(df.loc[best_index, "epoch"])

    final_row = df.iloc[-1]
    return {
        "num_epochs": int(df["epoch"].max()),
        "best_metric_name": metric_name,
        "best_metric_value": best_metric_value,
        "best_epoch": best_epoch,
        "final_epoch": int(final_row["epoch"]),
        "final_train_loss": float(final_row["train_loss"]) if pd.notna(final_row["train_loss"]) else None,
        "final_val_loss": float(final_row["val_loss"]) if pd.notna(final_row["val_loss"]) else None,
        "final_train_ave_f1": float(final_row["train_ave_f1"]) if pd.notna(final_row["train_ave_f1"]) else None,
        "final_val_ave_f1": float(final_row["val_ave_f1"]) if pd.notna(final_row["val_ave_f1"]) else None,
        "final_train_f1": float(final_row["train_f1"]) if pd.notna(final_row["train_f1"]) else None,
        "final_val_f1": float(final_row["val_f1"]) if pd.notna(final_row["val_f1"]) else None,
    }


def load_run_history(run_dir: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    log_path = run_dir / "run.log"
    df = parse_training_log(log_path)
    summary = summarize_history(df)
    return df, summary