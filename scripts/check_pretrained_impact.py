#!/usr/bin/env python3
"""Check whether pretrained weights required by enabled channels are present.

This script is intentionally lightweight: it does not train the model. It checks
configured paths and tries minimal imports/loads that reveal which channels are
blocked by missing pretrained weights.
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def size_mb(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    if path.is_dir():
        total = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
        return f"{total / 1024 / 1024:.2f} MB (dir total)"
    return f"{path.stat().st_size / 1024 / 1024:.2f} MB"


def status_line(label: str, path: str | Path) -> tuple[bool, str]:
    p = ROOT / path if not Path(path).is_absolute() else Path(path)
    ok = p.exists()
    mark = "OK" if ok else "MISSING"
    return ok, f"[{mark}] {label}: {p} | {size_mb(p)}"


def main() -> int:
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "configs/main_config.yaml"
    if not config_path.is_absolute():
        config_path = ROOT / config_path

    print("=== AdaLocML pretrained impact check ===")
    print(f"Project root: {ROOT}")
    print(f"Config: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    checks: list[tuple[bool, str]] = []

    # Files directly configured or hard-coded by the current pipeline.
    if cfg.get("rnaloclm", {}).get("enabled", False):
        checks.append(status_line("RNA-FM weight required by rnaloclm", cfg["rnaloclm"].get("rna_fm_model_path", "pretrained/RNA-FM/RNA-FM_pretrained.pth")))

    if cfg.get("ilocbert", {}).get("enabled", False):
        checks.append(status_line("DNABERT-2 directory required by ilocbert/data collate", "pretrained/DNABERT-2-117M"))
        checks.append(status_line("DNABERT-2 model.safetensors", "pretrained/DNABERT-2-117M/model.safetensors"))
        checks.append(status_line("DNABERT-2 config.json", "pretrained/DNABERT-2-117M/config.json"))
        checks.append(status_line("DNABERT-2 tokenizer config", "pretrained/DNABERT-2-117M/tokenizer_config.json"))

    if cfg.get("channel_agent", {}).get("enabled", False):
        checks.append(status_line("Channel agent checkpoint", cfg["channel_agent"].get("model_path", "pretrained/channel_agent.pth")))

    # Other important local data dependencies that can affect a smoke run.
    if cfg.get("cfploc", {}).get("enabled", False):
        checks.append(status_line("CGR feature dir", cfg["cfploc"].get("feature_dir", "data/cgr_features")))
    if cfg.get("intra_graph_channel", {}).get("enabled", False):
        checks.append(status_line("Structure CSV", cfg["intra_graph_channel"].get("structure_csv_path", "data/structures.csv")))
    if cfg.get("data", {}).get("csv_path"):
        checks.append(status_line("Main data CSV", cfg["data"]["csv_path"]))

    print("\n--- Path checks ---")
    missing = []
    for ok, line in checks:
        print(line)
        if not ok:
            missing.append(line)

    print("\n--- Minimal load checks ---")

    # DNABERT tokenizer load: this is what prepare_data() will do when ilocbert is enabled.
    if cfg.get("ilocbert", {}).get("enabled", False):
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(str(ROOT / "pretrained/DNABERT-2-117M"), trust_remote_code=True)
            print(f"[OK] DNABERT tokenizer load: {tok.__class__.__name__}")
        except Exception as exc:
            print(f"[FAIL] DNABERT tokenizer load: {type(exc).__name__}: {exc}")

    # RNA-FM import and local weight path check. Loading the full model may be slow/GPU-heavy,
    # so only perform it if --deep is passed.
    if cfg.get("rnaloclm", {}).get("enabled", False):
        try:
            import fm  # noqa: F401
            print("[OK] Python package 'fm' import")
        except Exception as exc:
            print(f"[FAIL] Python package 'fm' import: {type(exc).__name__}: {exc}")

        if "--deep" in sys.argv:
            try:
                import fm
                weight_path = str(ROOT / cfg["rnaloclm"].get("rna_fm_model_path", "pretrained/RNA-FM/RNA-FM_pretrained.pth"))
                fm.pretrained.rna_fm_t12(weight_path)
                print("[OK] RNA-FM full model load")
            except Exception as exc:
                print(f"[FAIL] RNA-FM full model load: {type(exc).__name__}: {exc}")
                traceback.print_exc(limit=3)

    print("\n--- Summary ---")
    if missing:
        print("Missing dependencies were found. With the current config, full training/inference will fail unless the affected channels are disabled or the files are restored.")
        return 2

    print("All configured pretrained/data paths checked by this script exist. If training still fails, run with --deep or run a one-batch smoke test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
