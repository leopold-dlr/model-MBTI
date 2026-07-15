"""Load run settings and the model portfolio from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    name: str
    provider: str
    model_id: str
    enabled: bool = True
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunSettings:
    n_runs: int = 10
    language: str = "en"
    mode: str = "one-shot"
    randomize_item_order: bool = True
    seed: int = 42
    max_retries: int = 3
    output_dir: str = "data/runs"
    report_dir: str = "reports"
    max_concurrency: int = 4


def load_run_settings(path: str | Path) -> RunSettings:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return RunSettings(
        n_runs=int(data.get("n_runs", 10)),
        language=str(data.get("language", "en")),
        mode=str(data.get("mode", "one-shot")),
        randomize_item_order=bool(data.get("randomize_item_order", True)),
        seed=int(data.get("seed", 42)),
        max_retries=int(data.get("max_retries", 3)),
        output_dir=str(data.get("output_dir", "data/runs")),
        report_dir=str(data.get("report_dir", "reports")),
        max_concurrency=int(data.get("max_concurrency", 4)),
    )


def load_models(path: str | Path) -> list[ModelConfig]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    models = []
    for entry in data.get("models", []):
        models.append(
            ModelConfig(
                name=str(entry["name"]),
                provider=str(entry["provider"]),
                model_id=str(entry["model_id"]),
                enabled=bool(entry.get("enabled", True)),
                params=dict(entry.get("params", {})),
            )
        )
    if not models:
        raise ValueError(f"No models defined in {path}.")
    names = [m.name for m in models]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate model names in models.yaml.")
    return models
