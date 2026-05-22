#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader, Subset

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import OUTPUTS_ROOT, ensure_parent_dir, safe_filename  # noqa: E402
from src.agent2.dynamic_fusion_model import DynamicFusionModel  # noqa: E402
from src.data.batch_collate import collate_fn  # noqa: E402
from src.features.lncmamba_utils import Tokenizer, lncRNA_loc_dataset  # noqa: E402
from src.models.meta_architect import MetaArchitect  # noqa: E402
from src.utils.helpers import setup_logging  # noqa: E402


logger = logging.getLogger("AMPLoc.utils_eval_attribution")

STRUCTURE_VIEW_CHANNELS = [
    "Struct-PB",
    "Struct-PC",
    "Struct-PE",
    "Struct-PBC",
    "Struct-PBE",
    "Struct-PCE",
    "Struct-PBCE",
]


def _normalize_key(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def _to_project_path(path_like: str | Path) -> Path:
    candidate = Path(path_like).expanduser()
    if candidate.is_absolute():
        return candidate

    repo_candidate = PROJECT_ROOT / candidate
    if repo_candidate.exists():
        return repo_candidate

    if candidate.exists():
        return candidate.resolve()

    return repo_candidate


def _deep_update(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            base[key] = _deep_update(dict(base[key]), value)
        else:
            base[key] = deepcopy(value)
    return base


def _slugify_label(label: str, index: int, used: set[str]) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(label)).strip("_").lower()
    if not slug:
        slug = f"label_{index}"

    candidate = slug
    suffix = 2
    while candidate in used:
        candidate = f"{slug}_{suffix}"
        suffix += 1

    used.add(candidate)
    return candidate


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _dataset_records(dataset: Any) -> list[dict[str, Any]]:
    return [dataset[index] for index in range(len(dataset))]


def _label_key(labels: Sequence[str]) -> str:
    return "_".join(sorted(str(label) for label in labels))


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    dataset_key: str
    config_path: Path
    development_csv: Path
    held_out_test_csv: Path

    def csv_for_split(self, split: str) -> Path:
        normalized = _normalize_key(split)
        if normalized in {"train", "dev", "development", "reference", "fit"}:
            return self.development_csv
        if normalized in {"test", "heldout", "heldouttest", "eval", "evaluation", "independenttest"}:
            return self.held_out_test_csv
        raise ValueError(f"Unsupported split '{split}' for benchmark '{self.name}'.")


def _build_benchmark_specs() -> dict[str, BenchmarkSpec]:
    dataset1 = BenchmarkSpec(
        name="Human-Lnc4",
        dataset_key="dataset1",
        config_path=PROJECT_ROOT / "configs" / "main_config.yaml",
        development_csv=PROJECT_ROOT / "data" / "rigorous_splits" / "dataset1_development.csv",
        held_out_test_csv=PROJECT_ROOT / "data" / "rigorous_splits" / "dataset1_held_out_test.csv",
    )
    dataset2 = BenchmarkSpec(
        name="Multi-Lnc7",
        dataset_key="dataset2",
        config_path=PROJECT_ROOT / "configs" / "main_config_dataset2_lr_scheduler.yaml",
        development_csv=PROJECT_ROOT / "data" / "rigorous_splits" / "dataset2_development.csv",
        held_out_test_csv=PROJECT_ROOT / "data" / "rigorous_splits" / "dataset2_held_out_test.csv",
    )

    return {
        "humanlnc4": dataset1,
        "dataset1": dataset1,
        "multilnc7": dataset2,
        "dataset2": dataset2,
    }


BENCHMARK_SPECS = _build_benchmark_specs()


@dataclass
class CheckpointInfo:
    path: Path
    epoch: int | None
    best_metric: float | None
    model_state_dict: dict[str, Any]
    optimizer_state_dict: dict[str, Any] | None
    agent_state_dict: dict[str, Any] | None
    manager_state_dict: dict[str, Any] | None
    metadata: dict[str, Any]
    missing_keys: list[str]
    unexpected_keys: list[str]


@dataclass
class EvaluationResult:
    y_true: np.ndarray
    y_prob: np.ndarray
    y_pred: np.ndarray
    sample_ids: list[Any]
    raw_sequences: list[str]
    label_names: list[str]
    threshold: float
    metrics: dict[str, Any]


@dataclass
class EvaluationBundle:
    config: dict[str, Any]
    reference_csv: Path | None
    eval_csv: Path | None
    reference_dataset: Any
    eval_dataset: Any
    tokenizer: Tokenizer
    mlb: MultiLabelBinarizer
    label_names: list[str]
    reference_loader: DataLoader
    eval_loader: DataLoader
    dnabert_tokenizer: Any | None


def resolve_benchmark_spec(name: str | None) -> BenchmarkSpec | None:
    if name is None:
        return None

    normalized = _normalize_key(name)
    spec = BENCHMARK_SPECS.get(normalized)
    if spec is None:
        raise ValueError(f"Unknown benchmark alias: {name}")
    return spec


def resolve_source_path(source: str | Path | None, *, split: str = "held_out_test") -> tuple[Path, BenchmarkSpec | None]:
    if source is None:
        raise ValueError("A source path or benchmark alias is required.")

    spec = resolve_benchmark_spec(str(source))
    if spec is not None:
        return spec.csv_for_split(split), spec

    path = Path(source).expanduser()
    if not path.is_absolute():
        repo_candidate = PROJECT_ROOT / path
        if repo_candidate.exists():
            return repo_candidate, None

    if path.exists():
        return path.resolve(), None

    raise FileNotFoundError(f"Source not found: {source}")


def resolve_config_path(source: str | Path | None) -> tuple[Path, BenchmarkSpec | None]:
    if source is None:
        return PROJECT_ROOT / "configs" / "main_config.yaml", None

    spec = resolve_benchmark_spec(str(source))
    if spec is not None:
        return spec.config_path, spec

    path = Path(source).expanduser()
    if not path.is_absolute():
        repo_candidate = PROJECT_ROOT / path
        if repo_candidate.exists():
            return repo_candidate, None

    if path.exists():
        return path.resolve(), None

    raise FileNotFoundError(f"Config file not found: {source}")


def load_config(config_source: str | Path | None = None, *, overrides: Mapping[str, Any] | None = None) -> tuple[dict[str, Any], BenchmarkSpec | None]:
    config_path, benchmark_spec = resolve_config_path(config_source)

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    if overrides:
        config = _deep_update(config, overrides)

    config = expand_active_fusion_channels(config)
    return config, benchmark_spec


def list_active_fusion_channels(config: Mapping[str, Any]) -> list[str]:
    active_channels: list[str] = []

    for channel_name in ["lncmamba", "rnaloclm", "cfploc", "ilocbert", "intra_graph_channel", "rpi_channel"]:
        channel_cfg = config.get(channel_name, {}) or {}
        if not channel_cfg.get("enabled", False):
            continue

        if channel_name == "intra_graph_channel":
            for view_name in STRUCTURE_VIEW_CHANNELS:
                if view_name not in active_channels:
                    active_channels.append(view_name)
        elif channel_name == "rpi_channel":
            for source_name in (channel_cfg.get("rpi_sources", {}) or {}).keys():
                if source_name not in active_channels:
                    active_channels.append(source_name)
        else:
            if channel_name not in active_channels:
                active_channels.append(channel_name)

    return active_channels


def expand_active_fusion_channels(config: Mapping[str, Any]) -> dict[str, Any]:
    expanded = deepcopy(dict(config))
    expanded.setdefault("meta_architect", {})
    expanded["meta_architect"]["active_fusion_channels"] = list_active_fusion_channels(expanded)
    logger.info("AMPLoc active fusion channels: %s", expanded["meta_architect"]["active_fusion_channels"])
    return expanded


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def select_device(preferred: str | torch.device | None = None) -> torch.device:
    if preferred is None or str(preferred).lower() == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(preferred)
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("Requested CUDA device is unavailable; falling back to CPU.")
        return torch.device("cpu")
    if device.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        logger.warning("Requested MPS device is unavailable; falling back to CPU.")
        return torch.device("cpu")
    return device


def resolve_structure_path(config: Mapping[str, Any], override: str | Path | None = None) -> Path | None:
    if override is not None:
        return _to_project_path(override)

    structure_cfg = config.get("intra_graph_channel", {}) or {}
    if structure_cfg.get("enabled", False):
        structure_path = structure_cfg.get("structure_csv_path")
        if structure_path:
            return _to_project_path(structure_path)

    return None


def load_dnabert_tokenizer(tokenizer_path: str | Path | None = None) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - dependency is expected to exist in this workspace
        raise ImportError("transformers is required for the iLoc-BERT tokenizer.") from exc

    path = _to_project_path(tokenizer_path or PROJECT_ROOT / "pretrained" / "DNABERT-2-117M")
    return AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)


def collect_dataset_records(dataset: Any) -> list[dict[str, Any]]:
    return _dataset_records(dataset)


def build_tokenizer_from_records(records: Sequence[Mapping[str, Any]], seq_max_len: int) -> Tokenizer:
    if not records:
        raise ValueError("Cannot build a tokenizer from an empty record set.")

    sequences = [record["sequence_kmers"] for record in records]
    labels = [record["labels_text"] for record in records]
    return Tokenizer(sequences, labels, seqMaxLen=seq_max_len)


def build_tokenizer_from_dataset(dataset: Any, seq_max_len: int) -> Tokenizer:
    return build_tokenizer_from_records(collect_dataset_records(dataset), seq_max_len)


def build_mlb_from_tokenizer(tokenizer: Tokenizer) -> MultiLabelBinarizer:
    mlb = MultiLabelBinarizer(classes=list(tokenizer.lab2id.values()))
    mlb.fit([[label_id] for label_id in tokenizer.lab2id.values()])
    return mlb


def build_label_names(tokenizer: Tokenizer) -> list[str]:
    return [tokenizer.id2lab[index] for index in range(tokenizer.labNum)]


def build_motif_token_ids(config: Mapping[str, Any], tokenizer: Tokenizer) -> list[int]:
    motifs = config.get("lncmamba", {}).get("motifs", []) or []
    return [tokenizer.tkn2id.get(motif, tokenizer.tkn2id.get("<UNK>", 0)) for motif in motifs]


def load_dataset(
    csv_path: str | Path,
    config: Mapping[str, Any],
    *,
    mode: str | None = None,
    structure_path: str | Path | None = None,
) -> lncRNA_loc_dataset:
    lncmamba_cfg = config.get("lncmamba", {}) or {}
    data_mode = mode or config.get("data", {}).get("mode", "normal")
    resolved_structure = resolve_structure_path(config, override=structure_path)
    return lncRNA_loc_dataset(
        dataPath=str(_to_project_path(csv_path)),
        k=int(lncmamba_cfg.get("k_mer", 3)),
        mode=data_mode,
        structure_path=str(resolved_structure) if resolved_structure is not None else None,
    )


def build_dataloader(
    dataset: Any,
    tokenizer: Tokenizer,
    mlb: MultiLabelBinarizer,
    config: Mapping[str, Any],
    *,
    batch_size: int | None = None,
    shuffle: bool = False,
    num_workers: int | None = None,
    dnabert_tokenizer: Any | None = None,
) -> DataLoader:
    training_cfg = config.get("training", {}) or {}
    resolved_batch_size = batch_size or int(training_cfg.get("batch_size", 32))
    resolved_num_workers = int(num_workers if num_workers is not None else training_cfg.get("num_workers", 0))

    collate = partial(
        collate_fn,
        tokenizer=tokenizer,
        mlb=mlb,
        config=config,
        dnabert_tokenizer=dnabert_tokenizer,
    )
    return DataLoader(
        dataset,
        batch_size=resolved_batch_size,
        shuffle=shuffle,
        num_workers=resolved_num_workers,
        collate_fn=collate,
    )


def prepare_evaluation_bundle_from_datasets(
    config: Mapping[str, Any],
    reference_dataset: Any,
    eval_dataset: Any,
    *,
    batch_size: int | None = None,
    num_workers: int | None = None,
    dnabert_tokenizer: Any | None = None,
) -> EvaluationBundle:
    reference_records = collect_dataset_records(reference_dataset)
    tokenizer = build_tokenizer_from_records(reference_records, int(config["lncmamba"]["seq_max_len"]))
    mlb = build_mlb_from_tokenizer(tokenizer)
    label_names = build_label_names(tokenizer)

    resolved_dnabert_tokenizer = dnabert_tokenizer
    if config.get("ilocbert", {}).get("enabled", False) and resolved_dnabert_tokenizer is None:
        resolved_dnabert_tokenizer = load_dnabert_tokenizer()

    reference_loader = build_dataloader(
        reference_dataset,
        tokenizer,
        mlb,
        config,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        dnabert_tokenizer=resolved_dnabert_tokenizer,
    )
    eval_loader = build_dataloader(
        eval_dataset,
        tokenizer,
        mlb,
        config,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        dnabert_tokenizer=resolved_dnabert_tokenizer,
    )

    return EvaluationBundle(
        config=dict(config),
        reference_csv=None,
        eval_csv=None,
        reference_dataset=reference_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        mlb=mlb,
        label_names=label_names,
        reference_loader=reference_loader,
        eval_loader=eval_loader,
        dnabert_tokenizer=resolved_dnabert_tokenizer,
    )


def prepare_evaluation_bundle(
    config: Mapping[str, Any],
    reference_csv: str | Path,
    eval_csv: str | Path,
    *,
    structure_path: str | Path | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    dnabert_tokenizer: Any | None = None,
) -> EvaluationBundle:
    reference_dataset = load_dataset(reference_csv, config, structure_path=structure_path)
    eval_dataset = load_dataset(eval_csv, config, structure_path=structure_path)
    bundle = prepare_evaluation_bundle_from_datasets(
        config,
        reference_dataset,
        eval_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        dnabert_tokenizer=dnabert_tokenizer,
    )
    return EvaluationBundle(
        config=bundle.config,
        reference_csv=_to_project_path(reference_csv),
        eval_csv=_to_project_path(eval_csv),
        reference_dataset=bundle.reference_dataset,
        eval_dataset=bundle.eval_dataset,
        tokenizer=bundle.tokenizer,
        mlb=bundle.mlb,
        label_names=bundle.label_names,
        reference_loader=bundle.reference_loader,
        eval_loader=bundle.eval_loader,
        dnabert_tokenizer=bundle.dnabert_tokenizer,
    )


def prepare_benchmark_evaluation_bundle(
    benchmark: str,
    config_source: str | Path | None = None,
    *,
    split: str = "held_out_test",
    structure_path: str | Path | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    dnabert_tokenizer: Any | None = None,
) -> EvaluationBundle:
    benchmark_spec = resolve_benchmark_spec(benchmark)
    config, _ = load_config(config_source or benchmark_spec.config_path)
    return prepare_evaluation_bundle(
        config,
        benchmark_spec.development_csv,
        benchmark_spec.csv_for_split(split),
        structure_path=structure_path,
        batch_size=batch_size,
        num_workers=num_workers,
        dnabert_tokenizer=dnabert_tokenizer,
    )


def build_stratified_holdout_indices(
    labels: Sequence[Sequence[str]],
    *,
    holdout_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    indices = np.arange(len(labels))
    stratify_keys = [_label_key(label_list) for label_list in labels]

    try:
        train_idx, holdout_idx = train_test_split(
            indices,
            test_size=holdout_ratio,
            random_state=seed,
            stratify=stratify_keys,
        )
    except ValueError:
        logger.warning("Stratified holdout split was not possible; falling back to a shuffled split.")
        train_idx, holdout_idx = train_test_split(
            indices,
            test_size=holdout_ratio,
            random_state=seed,
            shuffle=True,
            stratify=None,
        )

    return train_idx.tolist(), holdout_idx.tolist()


def build_stratified_kfold_splits(
    labels: Sequence[Sequence[str]],
    *,
    n_splits: int = 5,
    seed: int = 42,
) -> list[tuple[list[int], list[int]]]:
    indices = np.arange(len(labels))
    stratify_keys = [_label_key(label_list) for label_list in labels]
    class_counts = pd.Series(stratify_keys).value_counts()

    if len(class_counts) >= n_splits and int(class_counts.min()) >= n_splits:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return [(train_idx.tolist(), val_idx.tolist()) for train_idx, val_idx in splitter.split(indices, stratify_keys)]

    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    logger.warning("Falling back to non-stratified K-fold splits because some label groups are too small.")
    return [(train_idx.tolist(), val_idx.tolist()) for train_idx, val_idx in splitter.split(indices)]


def build_model(
    config: Mapping[str, Any],
    tokenizer: Tokenizer,
    device: torch.device,
    *,
    model_cls: type[torch.nn.Module] = DynamicFusionModel,
    project_root: str | Path = PROJECT_ROOT,
) -> torch.nn.Module:
    motif_tkn_ids = build_motif_token_ids(config, tokenizer)
    model = model_cls(config, tokenizer, motif_tkn_ids, device, str(project_root))
    return model


def _infer_model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _extract_state_dict(payload: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if isinstance(payload, Mapping):
        for key in ("model_state_dict", "state_dict", "model", "weights"):
            candidate = payload.get(key)
            if isinstance(candidate, Mapping):
                metadata = {str(k): v for k, v in payload.items() if k != key}
                return dict(candidate), metadata

        if all(hasattr(value, "shape") or torch.is_tensor(value) for value in payload.values()):
            return dict(payload), {}

    raise ValueError("The checkpoint does not contain a recognizable model state dictionary.")


def _strip_module_prefix(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    if not state_dict:
        return dict(state_dict)

    if any(key.startswith("module.") for key in state_dict.keys()):
        return {key[len("module."):]: value for key, value in state_dict.items()}

    return dict(state_dict)


def load_model_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device | None = None,
    strict: bool = True,
) -> CheckpointInfo:
    path = _to_project_path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    resolved_map_location = map_location or _infer_model_device(model)
    payload = torch.load(path, map_location=resolved_map_location)
    state_dict, metadata = _extract_state_dict(payload)
    state_dict = _strip_module_prefix(state_dict)

    load_result = model.load_state_dict(state_dict, strict=strict)
    missing_keys = list(getattr(load_result, "missing_keys", [])) if not strict else []
    unexpected_keys = list(getattr(load_result, "unexpected_keys", [])) if not strict else []

    optimizer_state = None
    if isinstance(payload, Mapping):
        optimizer_state = payload.get("optimizer_state_dict") if isinstance(payload.get("optimizer_state_dict"), Mapping) else None
        if optimizer is not None and optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)

    epoch_value = None
    if isinstance(payload, Mapping) and payload.get("epoch") is not None:
        epoch_value = int(payload["epoch"])

    best_metric = None
    if isinstance(payload, Mapping) and payload.get("best_metric") is not None:
        best_metric = float(payload["best_metric"])

    agent_state = None
    if isinstance(payload, Mapping) and isinstance(payload.get("agent_state_dict"), Mapping):
        agent_state = dict(payload["agent_state_dict"])

    manager_state = None
    if isinstance(payload, Mapping) and isinstance(payload.get("manager_state_dict"), Mapping):
        manager_state = dict(payload["manager_state_dict"])

    return CheckpointInfo(
        path=path,
        epoch=epoch_value,
        best_metric=best_metric,
        model_state_dict=state_dict,
        optimizer_state_dict=dict(optimizer_state) if optimizer_state is not None else None,
        agent_state_dict=agent_state,
        manager_state_dict=manager_state,
        metadata=metadata,
        missing_keys=missing_keys,
        unexpected_keys=unexpected_keys,
    )


def build_model_from_checkpoint(
    config: Mapping[str, Any],
    tokenizer: Tokenizer,
    device: torch.device,
    checkpoint_path: str | Path,
    *,
    model_cls: type[torch.nn.Module] = DynamicFusionModel,
    project_root: str | Path = PROJECT_ROOT,
    strict: bool = True,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[torch.nn.Module, CheckpointInfo]:
    model = build_model(config, tokenizer, device, model_cls=model_cls, project_root=project_root)
    info = load_model_checkpoint(model, checkpoint_path, optimizer=optimizer, map_location=device, strict=strict)
    return model, info


def extract_logits(outputs: Any) -> torch.Tensor:
    if isinstance(outputs, Mapping):
        for key in ("fused_logits", "logits", "pred_logits"):
            candidate = outputs.get(key)
            if torch.is_tensor(candidate):
                return candidate

        channel_logits = outputs.get("channel_logits")
        if isinstance(channel_logits, Mapping) and channel_logits:
            first_value = next(iter(channel_logits.values()))
            if torch.is_tensor(first_value):
                return first_value

    if torch.is_tensor(outputs):
        return outputs

    if isinstance(outputs, (list, tuple)):
        for item in outputs:
            if torch.is_tensor(item):
                return item
            if isinstance(item, Mapping):
                try:
                    return extract_logits(item)
                except (KeyError, TypeError, ValueError):
                    continue

    raise TypeError("Could not extract logits from the model output.")


def threshold_predictions(y_prob: np.ndarray | Sequence[Sequence[float]], threshold: float = 0.5) -> np.ndarray:
    y_prob_array = np.asarray(y_prob)
    if y_prob_array.ndim == 1:
        y_prob_array = y_prob_array[:, None]
    return (y_prob_array > threshold).astype(int)


def compute_per_label_f1(
    y_true: np.ndarray | Sequence[Sequence[int]],
    y_pred: np.ndarray | Sequence[Sequence[int]],
    label_names: Sequence[str] | None = None,
) -> dict[str, float]:
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    if y_true_array.ndim == 1:
        y_true_array = y_true_array[:, None]
    if y_pred_array.ndim == 1:
        y_pred_array = y_pred_array[:, None]

    per_label_scores = f1_score(y_true_array, y_pred_array, average=None, zero_division=0)
    if label_names is None:
        label_names = [f"label_{index}" for index in range(len(per_label_scores))]

    return {str(label_names[index]): float(score) for index, score in enumerate(per_label_scores)}


def compute_multilabel_metrics(
    y_true: np.ndarray | Sequence[Sequence[int]],
    y_prob: np.ndarray | Sequence[Sequence[float]],
    *,
    threshold: float = 0.5,
    label_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    y_true_array = np.asarray(y_true)
    y_prob_array = np.asarray(y_prob)

    if y_true_array.size == 0 or y_prob_array.size == 0:
        return {
            "Ave-F1": 0.0,
            "MiP": 0.0,
            "MiR": 0.0,
            "MiF": 0.0,
            "MaAUC": 0.0,
            "per_label_f1": {},
        }

    if y_true_array.ndim == 1:
        y_true_array = y_true_array[:, None]
    if y_prob_array.ndim == 1:
        y_prob_array = y_prob_array[:, None]

    y_pred_array = threshold_predictions(y_prob_array, threshold=threshold)

    ave_f1 = f1_score(y_true_array, y_pred_array, average="samples", zero_division=0)
    mip = precision_score(y_true_array, y_pred_array, average="micro", zero_division=0)
    mir = recall_score(y_true_array, y_pred_array, average="micro", zero_division=0)
    mif = f1_score(y_true_array, y_pred_array, average="micro", zero_division=0)

    auc_scores: list[float] = []
    for class_index in range(y_true_array.shape[1]):
        class_labels = y_true_array[:, class_index]
        if len(np.unique(class_labels)) > 1:
            auc_scores.append(float(roc_auc_score(class_labels, y_prob_array[:, class_index])))

    ma_auc = float(np.mean(auc_scores)) if auc_scores else 0.0

    return {
        "Ave-F1": float(ave_f1),
        "MiP": float(mip),
        "MiR": float(mir),
        "MiF": float(mif),
        "MaAUC": ma_auc,
        "per_label_f1": compute_per_label_f1(y_true_array, y_pred_array, label_names=label_names),
    }


def run_evaluation(
    model: torch.nn.Module,
    data_loader: DataLoader,
    *,
    threshold: float = 0.5,
    label_names: Sequence[str] | None = None,
) -> EvaluationResult:
    model.eval()

    all_y_true: list[np.ndarray] = []
    all_y_prob: list[np.ndarray] = []
    all_sample_ids: list[Any] = []
    all_raw_sequences: list[str] = []

    with torch.no_grad():
        for batch in data_loader:
            outputs = model(batch)
            logits = extract_logits(outputs)
            probabilities = torch.sigmoid(logits).detach().cpu().numpy()
            labels = batch["labels"].detach().cpu().numpy()

            all_y_prob.append(probabilities)
            all_y_true.append(labels)
            all_sample_ids.extend(batch.get("gene_ids", []))
            all_raw_sequences.extend(batch.get("raw_sequences", []))

    y_true = np.concatenate(all_y_true, axis=0) if all_y_true else np.zeros((0, 0), dtype=float)
    y_prob = np.concatenate(all_y_prob, axis=0) if all_y_prob else np.zeros((0, 0), dtype=float)
    y_pred = threshold_predictions(y_prob, threshold=threshold)
    resolved_label_names = list(label_names or [f"label_{index}" for index in range(y_true.shape[1] if y_true.ndim == 2 else 0)])
    metrics = compute_multilabel_metrics(y_true, y_prob, threshold=threshold, label_names=resolved_label_names)

    return EvaluationResult(
        y_true=y_true,
        y_prob=y_prob,
        y_pred=y_pred,
        sample_ids=all_sample_ids,
        raw_sequences=all_raw_sequences,
        label_names=resolved_label_names,
        threshold=threshold,
        metrics=metrics,
    )


def build_prediction_dataframe(result: EvaluationResult) -> pd.DataFrame:
    used_columns: set[str] = set()
    label_columns = [
        _slugify_label(label_name, index, used_columns)
        for index, label_name in enumerate(result.label_names)
    ]

    rows: list[dict[str, Any]] = []
    for sample_index, sample_id in enumerate(result.sample_ids):
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "true_labels": ";".join(
                result.label_names[label_index]
                for label_index in np.flatnonzero(result.y_true[sample_index]).tolist()
            ),
            "predicted_labels": ";".join(
                result.label_names[label_index]
                for label_index in np.flatnonzero(result.y_pred[sample_index]).tolist()
            ),
        }

        if sample_index < len(result.raw_sequences):
            row["raw_sequence"] = result.raw_sequences[sample_index]

        for label_index, column_name in enumerate(label_columns):
            row[f"true__{column_name}"] = int(result.y_true[sample_index, label_index])
            row[f"pred__{column_name}"] = int(result.y_pred[sample_index, label_index])
            row[f"prob__{column_name}"] = float(result.y_prob[sample_index, label_index])

        rows.append(row)

    return pd.DataFrame(rows)


def save_json_report(path: str | Path, payload: Any) -> Path:
    output_path = Path(path)
    ensure_parent_dir(output_path)
    output_path.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def save_prediction_table(
    result: EvaluationResult,
    output_path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    output_path = Path(output_path)
    ensure_parent_dir(output_path)

    table = build_prediction_dataframe(result)
    table.to_csv(output_path, index=False)

    used_columns: set[str] = set()
    label_columns = [
        _slugify_label(label_name, index, used_columns)
        for index, label_name in enumerate(result.label_names)
    ]
    metadata_payload = {
        "threshold": result.threshold,
        "label_names": result.label_names,
        "label_columns": {column_name: label_name for column_name, label_name in zip(label_columns, result.label_names)},
        "metrics": result.metrics,
    }
    if metadata:
        metadata_payload.update(metadata)

    save_json_report(output_path.with_suffix(".json"), metadata_payload)
    return output_path


def build_ig_baseline(
    batch: Mapping[str, Any],
    *,
    tokenizer: Tokenizer | None = None,
    dnabert_tokenizer: Any | None = None,
    tensor_strategy: str = "zeros",
    string_strategy: str = "keep",
) -> dict[str, Any]:
    baseline: dict[str, Any] = {}
    pad_token_id = tokenizer.tkn2id.get("<PAD>", 0) if tokenizer is not None else 0
    iloc_pad_token_id = getattr(dnabert_tokenizer, "pad_token_id", 0) or 0

    for key, value in batch.items():
        if torch.is_tensor(value):
            if tensor_strategy == "pad" and key == "input_ids":
                baseline[key] = torch.full_like(value, pad_token_id)
            elif tensor_strategy == "pad" and key == "iloc_input_ids":
                baseline[key] = torch.full_like(value, iloc_pad_token_id)
            elif key.endswith("attention_mask"):
                baseline[key] = torch.zeros_like(value)
            else:
                baseline[key] = torch.zeros_like(value)
            continue

        if isinstance(value, list):
            if string_strategy == "blank" and key in {"raw_sequences", "gene_ids"}:
                baseline[key] = ["" for _ in value]
            else:
                baseline[key] = list(value)
            continue

        baseline[key] = value

    return baseline


def build_holdout_split_from_dataset(
    dataset: Any,
    *,
    holdout_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[Subset, Subset]:
    records = collect_dataset_records(dataset)
    train_indices, holdout_indices = build_stratified_holdout_indices(
        [record["labels_text"] for record in records],
        holdout_ratio=holdout_ratio,
        seed=seed,
    )
    return Subset(dataset, train_indices), Subset(dataset, holdout_indices)


def main() -> None:
    parser = argparse.ArgumentParser(description="AMPLoc evaluation and attribution utility")
    parser.add_argument("--config", type=str, default=None, help="Config path or AMPLoc benchmark alias.")
    parser.add_argument("--benchmark", type=str, default=None, help="Human-Lnc4 or Multi-Lnc7.")
    parser.add_argument("--reference-csv", type=str, default=None, help="Reference split CSV used to build tokenizer.")
    parser.add_argument("--eval-csv", type=str, default=None, help="Evaluation split CSV.")
    parser.add_argument("--single-csv", type=str, default=None, help="Single CSV to split into 90/10 train and hold-out subsets.")
    parser.add_argument("--holdout-ratio", type=float, default=0.1, help="Hold-out ratio for --single-csv.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="auto", help="Device preference: auto, cpu, cuda, or mps.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to load before evaluation.")
    parser.add_argument("--model-class", type=str, default="dynamic", choices=("dynamic", "meta"), help="Model class to instantiate.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for multilabel predictions.")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional batch size override.")
    parser.add_argument("--num-workers", type=int, default=None, help="Optional dataloader worker override.")
    parser.add_argument("--structure-csv", type=str, default=None, help="Optional structure CSV override.")
    parser.add_argument("--dnabert-tokenizer", type=str, default=None, help="Optional DNABERT tokenizer path override.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory used to store AMPLoc outputs.")
    args = parser.parse_args()

    set_global_seed(args.seed)
    device = select_device(args.device)

    config_source = args.config or args.benchmark
    config, config_spec = load_config(config_source)
    benchmark_spec = resolve_benchmark_spec(args.benchmark) or config_spec

    output_dir = Path(args.output_dir) if args.output_dir else (OUTPUTS_ROOT / "amploc_analysis" / "utils_eval_attribution" / f"AMPLoc_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(str(output_dir / "AMPLoc_eval_attribution.log"))

    structure_path = resolve_structure_path(config, override=args.structure_csv)
    dnabert_tokenizer = load_dnabert_tokenizer(args.dnabert_tokenizer) if config.get("ilocbert", {}).get("enabled", False) else None

    if args.single_csv:
        full_dataset = load_dataset(args.single_csv, config, structure_path=structure_path)
        reference_subset, eval_subset = build_holdout_split_from_dataset(full_dataset, holdout_ratio=args.holdout_ratio, seed=args.seed)
        bundle = prepare_evaluation_bundle_from_datasets(
            config,
            reference_subset,
            eval_subset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            dnabert_tokenizer=dnabert_tokenizer,
        )
        bundle = EvaluationBundle(
            config=bundle.config,
            reference_csv=_to_project_path(args.single_csv),
            eval_csv=_to_project_path(args.single_csv),
            reference_dataset=reference_subset,
            eval_dataset=eval_subset,
            tokenizer=bundle.tokenizer,
            mlb=bundle.mlb,
            label_names=bundle.label_names,
            reference_loader=bundle.reference_loader,
            eval_loader=bundle.eval_loader,
            dnabert_tokenizer=bundle.dnabert_tokenizer,
        )
    else:
        if args.reference_csv is not None and args.eval_csv is not None:
            reference_csv = args.reference_csv
            eval_csv = args.eval_csv
        elif benchmark_spec is not None:
            reference_csv = benchmark_spec.development_csv
            eval_csv = benchmark_spec.csv_for_split("held_out_test")
        else:
            raise ValueError("Provide either --single-csv, a benchmark alias, or both --reference-csv and --eval-csv.")

        bundle = prepare_evaluation_bundle(
            config,
            reference_csv,
            eval_csv,
            structure_path=structure_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            dnabert_tokenizer=dnabert_tokenizer,
        )

    model_cls = DynamicFusionModel if args.model_class == "dynamic" else MetaArchitect
    model = build_model(config, bundle.tokenizer, device, model_cls=model_cls, project_root=PROJECT_ROOT)

    checkpoint_info = None
    if args.checkpoint:
        checkpoint_info = load_model_checkpoint(model, args.checkpoint, map_location=device, strict=True)
        logger.info("AMPLoc checkpoint loaded from %s", checkpoint_info.path)

    result = run_evaluation(model, bundle.eval_loader, threshold=args.threshold, label_names=bundle.label_names)
    metrics_path = save_json_report(output_dir / "AMPLoc_metrics.json", result.metrics)
    predictions_path = save_prediction_table(
        result,
        output_dir / "AMPLoc_predictions.csv",
        metadata={
            "config_source": str(config_source) if config_source is not None else None,
            "benchmark": benchmark_spec.name if benchmark_spec is not None else None,
            "checkpoint": str(checkpoint_info.path) if checkpoint_info is not None else None,
            "reference_csv": str(bundle.reference_csv),
            "eval_csv": str(bundle.eval_csv),
        },
    )

    logger.info("AMPLoc metrics written to %s", metrics_path)
    logger.info("AMPLoc predictions written to %s", predictions_path)
    print(json.dumps(_jsonable(result.metrics), indent=2, ensure_ascii=False))


__all__ = [
    "BenchmarkSpec",
    "CheckpointInfo",
    "EvaluationBundle",
    "EvaluationResult",
    "BENCHMARK_SPECS",
    "STRUCTURE_VIEW_CHANNELS",
    "build_holdout_split_from_dataset",
    "build_ig_baseline",
    "build_label_names",
    "build_dataloader",
    "build_dataset",  # Backward-compatible alias-like name exported below
    "build_mlb_from_tokenizer",
    "build_model",
    "build_model_from_checkpoint",
    "build_motif_token_ids",
    "build_per_label_f1",  # alias exported below
    "build_prediction_dataframe",
    "build_stratified_holdout_indices",
    "build_stratified_kfold_splits",
    "build_tokenizer_from_dataset",
    "build_tokenizer_from_records",
    "collect_dataset_records",
    "compute_multilabel_metrics",
    "compute_per_label_f1",
    "expand_active_fusion_channels",
    "extract_logits",
    "load_config",
    "load_dataset",
    "load_dnabert_tokenizer",
    "load_model_checkpoint",
    "list_active_fusion_channels",
    "prepare_benchmark_evaluation_bundle",
    "prepare_evaluation_bundle",
    "prepare_evaluation_bundle_from_datasets",
    "resolve_benchmark_spec",
    "resolve_source_path",
    "run_evaluation",
    "save_json_report",
    "save_prediction_table",
    "select_device",
    "set_global_seed",
    "threshold_predictions",
]


# Backward-compatible aliases for callers that prefer direct dataset terminology.
build_dataset = load_dataset
build_per_label_f1 = compute_per_label_f1


if __name__ == "__main__":
    main()