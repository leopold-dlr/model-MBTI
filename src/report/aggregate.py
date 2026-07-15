"""Aggregate raw run records into per-model statistics.

The headline statistic is *stability*: across N runs, how often does a model
land on the same type / same letter per axis? A model with low variance has a
consistent "personality"; a model that flips between types run-to-run is itself
an interesting result.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


def load_records(runs_dir: str | Path) -> list[dict]:
    records = []
    for path in sorted(Path(runs_dir).glob("*.json")):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return records


@dataclass
class AxisStats:
    axis: str
    pole_low: str
    pole_high: str
    modal_letter: str
    modal_freq: float            # fraction of valid runs on the modal letter
    letter_counts: dict[str, int]
    mean_pct_high: float         # mean preference toward pole_high (0-100)
    std_pct_high: float          # std dev of that preference across runs
    mean_raw_sum: float

    def to_dict(self) -> dict:
        return {
            "axis": self.axis,
            "pole_low": self.pole_low,
            "pole_high": self.pole_high,
            "modal_letter": self.modal_letter,
            "modal_freq": round(self.modal_freq, 4),
            "letter_counts": self.letter_counts,
            "mean_pct_high": round(self.mean_pct_high, 2),
            "std_pct_high": round(self.std_pct_high, 2),
            "mean_raw_sum": round(self.mean_raw_sum, 2),
        }


@dataclass
class ModelStats:
    model_name: str
    provider: str
    model_id: str
    n_total: int
    n_valid: int
    n_invalid: int
    modal_type: str
    modal_type_freq: float       # stability of the full 4-letter type
    type_counts: dict[str, int]
    axes: dict[str, AxisStats] = field(default_factory=dict)
    invalid_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "provider": self.provider,
            "model_id": self.model_id,
            "n_total": self.n_total,
            "n_valid": self.n_valid,
            "n_invalid": self.n_invalid,
            "modal_type": self.modal_type,
            "modal_type_freq": round(self.modal_type_freq, 4),
            "type_counts": self.type_counts,
            "axes": {k: v.to_dict() for k, v in self.axes.items()},
            "invalid_reasons": self.invalid_reasons,
        }


def _std(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def aggregate_model(model_name: str, records: list[dict]) -> ModelStats:
    recs = [r for r in records if r.get("model_name") == model_name]
    valid = [r for r in recs if r.get("valid")]
    invalid = [r for r in recs if not r.get("valid")]

    provider = recs[0].get("provider", "") if recs else ""
    model_id = recs[0].get("model_id", "") if recs else ""

    type_counts = Counter(r["score"]["type"] for r in valid)
    modal_type, modal_type_count = ("N/A", 0)
    if type_counts:
        modal_type, modal_type_count = type_counts.most_common(1)[0]
    modal_type_freq = (modal_type_count / len(valid)) if valid else 0.0

    # Collect per-axis series.
    axes: dict[str, AxisStats] = {}
    if valid:
        axis_names = list(valid[0]["score"]["axes"].keys())
        for axis in axis_names:
            letters = [r["score"]["axes"][axis]["letter"] for r in valid]
            pct_high = [float(r["score"]["axes"][axis]["pct_high"]) for r in valid]
            raw_sum = [float(r["score"]["axes"][axis]["raw_sum"]) for r in valid]
            pole_low = valid[0]["score"]["axes"][axis]["pole_low"]
            pole_high = valid[0]["score"]["axes"][axis]["pole_high"]
            letter_counts = dict(Counter(letters))
            modal_letter, modal_count = Counter(letters).most_common(1)[0]
            axes[axis] = AxisStats(
                axis=axis,
                pole_low=pole_low,
                pole_high=pole_high,
                modal_letter=modal_letter,
                modal_freq=modal_count / len(valid),
                letter_counts=letter_counts,
                mean_pct_high=statistics.fmean(pct_high),
                std_pct_high=_std(pct_high),
                mean_raw_sum=statistics.fmean(raw_sum),
            )

    return ModelStats(
        model_name=model_name,
        provider=provider,
        model_id=model_id,
        n_total=len(recs),
        n_valid=len(valid),
        n_invalid=len(invalid),
        modal_type=modal_type,
        modal_type_freq=modal_type_freq,
        type_counts=dict(type_counts),
        axes=axes,
        invalid_reasons=[r.get("invalid_reason", "unknown") for r in invalid],
    )


def aggregate_all(records: list[dict]) -> list[ModelStats]:
    names: list[str] = []
    for r in records:
        n = r.get("model_name")
        if n and n not in names:
            names.append(n)
    return [aggregate_model(n, records) for n in names]
