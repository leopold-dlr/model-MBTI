"""Aggregate raw run records into per-(model, condition) statistics.

The headline statistic is *stability*: across N runs, how often does a model
land on the same type / same letter per axis? A model with low variance has a
consistent "personality"; a model that flips between types run-to-run is
itself an interesting result. Because temperature is a confound for that
comparison (higher sampling entropy alone produces more flipping, independent
of any real trait), stats are grouped separately per temperature_condition
(and per prompt_variant, for the optional wording ablation) rather than
pooled -- see run_settings.yaml.
"""

from __future__ import annotations

import json
import math
import random
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from ..instrument import Instrument
from ..scoring.mbti_scorer import score_answers

Z_95 = 1.959963984540054  # two-sided 95% normal quantile


def load_records(runs_dir: str | Path) -> list[dict]:
    records = []
    for path in sorted(Path(runs_dir).glob("*.json")):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return records


def list_experiment_ids(records: list[dict]) -> list[str]:
    """All experiment ids present, oldest first. Records with none (legacy /
    pre-experiment_id schema) are not represented here."""
    return sorted({r["experiment_id"] for r in records if r.get("experiment_id")})


def latest_experiment_id(records: list[dict]) -> str | None:
    ids = list_experiment_ids(records)
    return ids[-1] if ids else None


def filter_experiment(records: list[dict], experiment_id: str | None) -> list[dict]:
    """Keep only records from one experiment (default: the most recent one),
    so two separate launches (e.g. a cheap smoke test, then the real run) are
    never silently pooled into the same stability figures. Records predating
    the experiment_id field (none present at all) are passed through as-is."""
    if not list_experiment_ids(records):
        return records
    if experiment_id is None:
        experiment_id = latest_experiment_id(records)
    return [r for r in records if r.get("experiment_id") == experiment_id]


def wilson_ci(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion, as (lo, hi) on a
    0-1 scale. Preferred over the normal approximation here because n is small
    (<=20 per condition) and many observed proportions sit near 0 or 1, where
    the normal approximation is known to misbehave (can even fall outside
    [0,1])."""
    if n == 0:
        return (0.0, 1.0)
    phat = successes / n
    denom = 1 + z * z / n
    center = phat + z * z / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    lo = (center - margin) / denom
    hi = (center + margin) / denom
    return (max(0.0, lo), min(1.0, hi))


def cronbach_alpha(item_matrix: list[list[float]]) -> float | None:
    """Cronbach's alpha for a (runs x items) oriented-score matrix from one
    axis. None if undefined (fewer than 2 items, fewer than 2 runs, or zero
    total variance). A low/negative alpha means this axis's items do not
    covary for this model -- i.e. the axis letter is not measuring one
    coherent thing, whatever its apparent stability."""
    if len(item_matrix) < 2:
        return None
    n_items = len(item_matrix[0])
    if n_items < 2:
        return None
    item_variances = [statistics.pvariance([row[j] for row in item_matrix]) for j in range(n_items)]
    totals = [sum(row) for row in item_matrix]
    total_variance = statistics.pvariance(totals)
    if total_variance == 0:
        return None
    return (n_items / (n_items - 1)) * (1 - sum(item_variances) / total_variance)


def _std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


@dataclass
class AxisStats:
    axis: str
    pole_low: str
    pole_high: str
    modal_letter: str
    modal_freq: float                     # fraction of valid runs on the modal letter
    modal_freq_ci: tuple[float, float]     # 95% Wilson CI on modal_freq
    letter_counts: dict[str, int]
    mean_pct_high: float                  # mean preference toward pole_high (0-100)
    std_pct_high: float                   # std dev of that preference across runs
    mean_raw_sum: float
    dist_from_midpoint: float             # |mean_pct_high - 50|, 0-50; 0 = pure coin flip
    cronbach_alpha: float | None = None    # None if undefined (see cronbach_alpha())
    trait_low: str | None = None          # human-readable trait name for pole_low
    trait_high: str | None = None         # human-readable trait name for pole_high

    @property
    def modal_trait(self) -> str:
        """The modal letter's human-readable trait name, falling back to the
        bare letter if the instrument doesn't define trait_low/trait_high."""
        name = self.trait_high if self.modal_letter == self.pole_high else self.trait_low
        return name or self.modal_letter

    def to_dict(self) -> dict:
        return {
            "axis": self.axis,
            "pole_low": self.pole_low,
            "pole_high": self.pole_high,
            "trait_low": self.trait_low,
            "trait_high": self.trait_high,
            "modal_letter": self.modal_letter,
            "modal_trait": self.modal_trait,
            "modal_freq": round(self.modal_freq, 4),
            "modal_freq_ci": [round(v, 4) for v in self.modal_freq_ci],
            "letter_counts": self.letter_counts,
            "mean_pct_high": round(self.mean_pct_high, 2),
            "std_pct_high": round(self.std_pct_high, 2),
            "mean_raw_sum": round(self.mean_raw_sum, 2),
            "dist_from_midpoint": round(self.dist_from_midpoint, 2),
            "cronbach_alpha": round(self.cronbach_alpha, 3) if self.cronbach_alpha is not None else None,
        }


@dataclass
class ModelStats:
    model_name: str
    provider: str
    model_id: str
    temperature_condition: str
    prompt_variant: str
    n_total: int
    n_valid: int
    n_invalid: int
    modal_type: str
    modal_type_freq: float                    # stability of the full 4-letter type
    modal_type_freq_ci: tuple[float, float]    # 95% Wilson CI on modal_type_freq
    type_counts: dict[str, int]
    reliable: bool                             # n_valid >= min_valid_runs
    pct_first_attempt: float                   # share of valid runs answered with no retry
    axes: dict[str, AxisStats] = field(default_factory=dict)
    invalid_reasons: list[str] = field(default_factory=list)
    example_items: list[dict] = field(default_factory=list)  # see item_level_examples()

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "provider": self.provider,
            "model_id": self.model_id,
            "temperature_condition": self.temperature_condition,
            "prompt_variant": self.prompt_variant,
            "n_total": self.n_total,
            "n_valid": self.n_valid,
            "n_invalid": self.n_invalid,
            "example_items": self.example_items,
            "modal_type": self.modal_type,
            "modal_type_freq": round(self.modal_type_freq, 4),
            "modal_type_freq_ci": [round(v, 4) for v in self.modal_type_freq_ci],
            "type_counts": self.type_counts,
            "reliable": self.reliable,
            "pct_first_attempt": round(self.pct_first_attempt, 4),
            "axes": {k: v.to_dict() for k, v in self.axes.items()},
            "invalid_reasons": self.invalid_reasons,
        }


def _axis_item_matrix(
    inst: Instrument, axis: str, valid_records: list[dict]
) -> list[list[float]]:
    """(runs x items) oriented-score matrix for one axis, honoring each run's
    own item_polarity (polarity is counterbalanced per run -- see
    prompting.templates.flip_map -- so the *effective* keyed direction can
    differ run to run for the same item)."""
    items = inst.items_for_axis(axis)
    if len(items) < 2:
        return []
    matrix: list[list[float]] = []
    for r in valid_records:
        answers = r.get("answers")
        if not answers:
            continue
        polarity = r.get("item_polarity", {})
        row: list[float] = []
        for it in items:
            raw = answers.get(str(it.id))
            if raw is None:
                break
            keyed = it.keyed
            if polarity.get(str(it.id)) == -1:
                keyed = -keyed
            oriented = raw if keyed == 1 else (inst.scale_min + inst.scale_max) - raw
            row.append(float(oriented))
        else:
            matrix.append(row)
    return matrix


def _describe_item_lean(left: str, right: str, toward_right: bool) -> str:
    """A human-readable phrase for which way an item leaned.

    Detects the "same base statement, different accuracy qualifier" pattern
    used to administer unipolar Likert instruments (e.g. IPIP-50) through
    this project's bipolar left/right schema (see ipip50_bigfive.yaml's
    header comment: "<qualifier>: <statement>", same statement on both
    sides) by comparing the text after the first ": " -- NOT a raw
    common-suffix match, which is fooled by "inaccurate" containing
    "accurate" as a substring and would chop the qualifier itself into the
    "shared" text. Phrases the unipolar case as a single accuracy judgment
    ("called X accurate of itself") rather than a confusing choice between
    two near-identical-looking strings.
    """
    left_head, _, left_rest = left.partition(": ")
    right_head, _, right_rest = right.partition(": ")
    if left_rest and right_rest and left_rest == right_rest:
        statement = left_rest.strip().rstrip(".").strip()
        verdict = "accurate" if toward_right else "inaccurate"
        return f'called "{statement}" **{verdict}** of itself'
    chosen = right if toward_right else left
    other = left if toward_right else right
    return f'leaned toward **"{chosen}"** over "{other}"'


def item_level_examples(
    inst: Instrument, valid_records: list[dict], n: int = 3
) -> list[dict]:
    """The model's most decisive and consistent individual answers -- concrete
    grounding for "what did the model actually say", not just aggregate axis
    stats. For each item, compute the oriented score (toward the item's
    canonical RIGHT anchor, honoring each run's own polarity flip so a
    flipped-this-run answer is comparable to an unflipped one) across valid
    runs, then rank items by how decisive (far from the midpoint) AND
    consistent (low spread) the answer was, descending.
    """
    if not valid_records:
        return []
    midpoint = inst.scale_midpoint
    span = inst.scale_max - inst.scale_min
    scored: list[dict] = []
    for it in inst.items:
        oriented: list[float] = []
        for r in valid_records:
            answers = r.get("answers")
            if not answers:
                continue
            raw = answers.get(str(it.id))
            if raw is None:
                continue
            flipped = r.get("item_polarity", {}).get(str(it.id)) == -1
            oriented.append(float((inst.scale_min + inst.scale_max - raw) if flipped else raw))
        if not oriented:
            continue
        mean = statistics.fmean(oriented)
        std = _std(oriented)
        decisiveness = abs(mean - midpoint) / (span / 2) * 100  # 0-100
        toward_right = mean > midpoint
        pct = (
            100.0 * (mean - inst.scale_min) / span if span else 50.0
        )
        scored.append(
            {
                "item_id": it.id,
                "axis": it.axis,
                "left": it.left,
                "right": it.right,
                "leans_toward": it.right if toward_right else it.left,
                "other_side": it.left if toward_right else it.right,
                "phrase": _describe_item_lean(it.left, it.right, toward_right),
                "pct": round(pct if toward_right else 100 - pct, 1),
                "mean": round(mean, 2),
                "std": round(std, 2),
                "n": len(oriented),
                # Higher = clearer signal: far from the midpoint AND low spread.
                "salience": round(decisiveness - std * 15, 2),
            }
        )
    scored.sort(key=lambda d: d["salience"], reverse=True)
    return scored[:n]


def aggregate_group(
    key: tuple[str, str, str],
    records: list[dict],
    inst: Instrument | None,
    min_valid_runs: int,
) -> ModelStats:
    model_name, temperature_condition, prompt_variant = key
    recs = [
        r
        for r in records
        if r.get("model_name") == model_name
        and r.get("temperature_condition", "default") == temperature_condition
        and r.get("prompt_variant", "default") == prompt_variant
    ]
    valid = [r for r in recs if r.get("valid")]
    invalid = [r for r in recs if not r.get("valid")]

    provider = recs[0].get("provider", "") if recs else ""
    model_id = recs[0].get("model_id", "") if recs else ""

    type_counts = Counter(r["score"]["type"] for r in valid)
    modal_type, modal_type_count = ("N/A", 0)
    if type_counts:
        modal_type, modal_type_count = type_counts.most_common(1)[0]
    modal_type_freq = (modal_type_count / len(valid)) if valid else 0.0
    modal_type_freq_ci = wilson_ci(modal_type_count, len(valid)) if valid else (0.0, 1.0)

    first_attempt = sum(1 for r in valid if r.get("succeeded_at_attempt") == 0)
    pct_first_attempt = (first_attempt / len(valid)) if valid else 0.0

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
            mean_pct_high = statistics.fmean(pct_high)

            alpha = None
            trait_low = trait_high = None
            if inst is not None:
                matrix = _axis_item_matrix(inst, axis, valid)
                if matrix:
                    alpha = cronbach_alpha(matrix)
                axis_meta = inst.axes.get(axis, {})
                trait_low = axis_meta.get("trait_low")
                trait_high = axis_meta.get("trait_high")

            axes[axis] = AxisStats(
                axis=axis,
                pole_low=pole_low,
                pole_high=pole_high,
                trait_low=trait_low,
                trait_high=trait_high,
                modal_letter=modal_letter,
                modal_freq=modal_count / len(valid),
                modal_freq_ci=wilson_ci(modal_count, len(valid)),
                letter_counts=letter_counts,
                mean_pct_high=mean_pct_high,
                std_pct_high=_std(pct_high),
                mean_raw_sum=statistics.fmean(raw_sum),
                dist_from_midpoint=abs(mean_pct_high - 50.0),
                cronbach_alpha=alpha,
            )

    examples = item_level_examples(inst, valid) if inst is not None else []

    return ModelStats(
        model_name=model_name,
        provider=provider,
        model_id=model_id,
        temperature_condition=temperature_condition,
        prompt_variant=prompt_variant,
        n_total=len(recs),
        n_valid=len(valid),
        n_invalid=len(invalid),
        modal_type=modal_type,
        modal_type_freq=modal_type_freq,
        modal_type_freq_ci=modal_type_freq_ci,
        type_counts=dict(type_counts),
        reliable=len(valid) >= min_valid_runs,
        pct_first_attempt=pct_first_attempt,
        axes=axes,
        invalid_reasons=[r.get("invalid_reason", "unknown") for r in invalid],
        example_items=examples,
    )


def aggregate_all(
    records: list[dict],
    inst: Instrument | None = None,
    experiment_id: str | None = None,
    min_valid_runs: int = 10,
) -> list[ModelStats]:
    """Group records by (model_name, temperature_condition, prompt_variant)
    within one experiment (the latest, unless `experiment_id` is given) and
    compute stats for each group.

    `inst` is optional but required to get Cronbach's alpha (needs the
    instrument's axis/item/keying metadata plus each run's raw per-item
    answers); without it, alpha is left as None everywhere.
    """
    records = filter_experiment(records, experiment_id)
    keys: list[tuple[str, str, str]] = []
    for r in records:
        name = r.get("model_name")
        if not name:
            continue
        key = (name, r.get("temperature_condition", "default"), r.get("prompt_variant", "default"))
        if key not in keys:
            keys.append(key)
    return [aggregate_group(k, records, inst, min_valid_runs) for k in keys]


def random_baseline_stats(
    inst: Instrument, n_runs: int, n_trials: int = 500, seed: int = 12345
) -> dict:
    """Monte Carlo reference point: the type/axis stability a purely random
    responder (uniform integer answers, no signal at all) would show over
    `n_trials` independent simulated "models" of `n_runs` runs each.

    A model whose measured stability is not clearly above this baseline is
    not demonstrating any detectable response consistency, whatever its modal
    type looks like -- this number belongs on every stability chart as a
    floor, not just in a footnote.
    """
    rng = random.Random(seed)
    type_freqs: list[float] = []
    axis_freqs: dict[str, list[float]] = {axis: [] for axis in inst.type_order}

    for _ in range(n_trials):
        results = []
        for _run in range(n_runs):
            answers = {it.id: rng.randint(inst.scale_min, inst.scale_max) for it in inst.items}
            results.append(score_answers(inst, answers))
        types = Counter(res.type for res in results)
        _, modal_count = types.most_common(1)[0]
        type_freqs.append(modal_count / n_runs)
        for axis in inst.type_order:
            letters = Counter(res.axes[axis].letter for res in results)
            _, modal_count_axis = letters.most_common(1)[0]
            axis_freqs[axis].append(modal_count_axis / n_runs)

    type_freqs.sort()
    return {
        "n_trials": n_trials,
        "n_runs": n_runs,
        "modal_type_freq_mean": statistics.fmean(type_freqs),
        "modal_type_freq_p05": _percentile(type_freqs, 0.05),
        "modal_type_freq_p95": _percentile(type_freqs, 0.95),
        "axes": {axis: statistics.fmean(vals) for axis, vals in axis_freqs.items()},
    }
