"""Drive the whole experiment: for each model x condition x N runs, prompt ->
parse -> score -> persist a raw run file.

Every raw run is stored in full (both prompts sent on every attempt, raw
reply, parsed answers, score, usage) so the scoring can be replayed later
without re-hitting the APIs, and so refusals/invalid runs remain auditable.
"""

from __future__ import annotations

import hashlib
import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from ..config import ModelConfig, RunSettings, TemperatureCondition
from ..instrument import Instrument
from ..prompting import templates
from ..prompting.parser import ParseError, parse_answers
from ..providers.base import ProviderError
from ..providers.registry import build_adapter
from ..scoring.mbti_scorer import score_answers

SCHEMA_VERSION = 2


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_experiment_id() -> str:
    """A fresh id stamped once per `run_experiment` invocation, so runs from
    different launches are never silently pooled together by the aggregator
    (and so the same command can be replayed later to track drift over
    time)."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _run_seed(base_seed: int, model_name: str, run_index: int) -> int:
    """Deterministic per-run seed so item order/polarity is reproducible.

    Uses SHA-256 rather than the builtin ``hash()``: Python salts string
    hashes randomly per-process (PYTHONHASHSEED) unless disabled, so
    ``hash((base_seed, model_name, run_index))`` gives a DIFFERENT seed every
    time the process restarts -- silently breaking the "reproducible item
    order" guarantee this project advertises. SHA-256 of a fixed string
    representation is stable across processes, machines, and Python
    versions.
    """
    key = f"{base_seed}:{model_name}:{run_index}".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    return int(digest, 16) & 0x7FFFFFFF


def execute_single_run(
    model: ModelConfig,
    inst: Instrument,
    settings: RunSettings,
    run_index: int,
    temperature_condition: TemperatureCondition,
    prompt_variant: str,
) -> dict:
    """Execute one run (with retries). Returns a fully-populated run record.

    Two independent retry budgets are tracked:

    * ``content_attempt`` -- the reply came back but was refused / malformed
      JSON. Retried up to ``settings.max_retries`` with an explicit reminder
      appended to the system prompt (this is itself part of the measurement:
      does the model comply under pressure?).
    * ``provider_attempt`` -- the API call itself failed (network error, rate
      limit, 5xx). Retried up to ``settings.max_provider_retries`` with
      exponential backoff and NO reminder -- this is infrastructure noise, not
      a model behavior, and must not be conflated with a content refusal.
    """
    adapter = build_adapter(model.provider, model.model_id, model.params)
    seed = _run_seed(settings.seed, model.name, run_index)
    randomize = settings.randomize_item_order
    flips = templates.flip_map(inst, seed)
    keyed_overrides = templates.keyed_overrides_from_flips(inst, flips)

    call_overrides: dict = {}
    if temperature_condition.value is not None:
        call_overrides["temperature"] = temperature_condition.value

    record: dict = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": _now_iso(),
        "model_name": model.name,
        "provider": model.provider,
        "model_id": model.model_id,
        "instrument": {"name": inst.name, "version": inst.version},
        "run_index": run_index,
        "temperature_condition": temperature_condition.label,
        "temperature_value": temperature_condition.value,
        "prompt_variant": prompt_variant,
        "call_params": adapter.merged_params(**call_overrides),
        "seed": seed,
        "item_polarity": {str(k): v for k, v in flips.items()},
        "language": settings.language,
        "mode": settings.mode,
        "valid": False,
        "succeeded_at_attempt": None,
        "attempts": [],
    }

    content_attempt = 0
    provider_attempt = 0
    total_attempt = 0
    last_error = None

    while True:
        is_retry = content_attempt > 0
        system, user, id_order = templates.build_prompt(
            inst,
            seed=seed,
            randomize=randomize,
            retry=is_retry,
            flips=flips,
            prompt_variant=prompt_variant,
        )
        attempt_log: dict = {
            "attempt": total_attempt,
            "content_attempt": content_attempt,
            "provider_attempt": provider_attempt,
            "is_retry": is_retry,
            "system_prompt": system,
            "user_prompt": user,
            "item_order": id_order,
        }
        try:
            gen = adapter.generate(system, user, **call_overrides)
        except ProviderError as exc:
            attempt_log["error"] = f"provider_error: {exc}"
            record["attempts"].append(attempt_log)
            last_error = str(exc)
            provider_attempt += 1
            total_attempt += 1
            if provider_attempt > settings.max_provider_retries:
                break
            time.sleep(min(2**provider_attempt, 30))
            continue

        attempt_log["raw_response"] = gen.text
        attempt_log["usage"] = gen.usage
        attempt_log["returned_model"] = gen.model

        try:
            answers = parse_answers(gen.text, id_order, inst.scale_min, inst.scale_max)
        except ParseError as exc:
            attempt_log["error"] = f"parse_error: {exc}"
            record["attempts"].append(attempt_log)
            last_error = str(exc)
            content_attempt += 1
            total_attempt += 1
            if content_attempt > settings.max_retries:
                break
            continue

        # Success: score it.
        result = score_answers(inst, answers, keyed_overrides=keyed_overrides)
        attempt_log["answers"] = {str(k): v for k, v in answers.items()}
        record["attempts"].append(attempt_log)
        record["valid"] = True
        record["succeeded_at_attempt"] = total_attempt
        record["answers"] = {str(k): v for k, v in answers.items()}
        record["score"] = result.to_dict()
        record["item_order"] = id_order
        record["usage"] = gen.usage
        record["returned_model"] = gen.model
        return record

    record["invalid_reason"] = last_error or "unknown"
    return record


def _write_record(record: dict, out_dir: Path) -> Path:
    ts = record["timestamp"].replace(":", "").replace("-", "").replace(".", "_")
    cond = record.get("temperature_condition", "default")
    variant = record.get("prompt_variant", "default")
    fname = (
        f"{record['model_name']}__{cond}__{variant}"
        f"__run{record['run_index']:02d}__{ts}.json"
    )
    path = out_dir / fname
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def run_experiment(
    models: list[ModelConfig],
    inst: Instrument,
    settings: RunSettings,
    only: list[str] | None = None,
    resume_experiment_id: str | None = None,
    log=print,
) -> list[dict]:
    """Run every enabled (and available) model x condition x n_runs. Returns
    all records.

    ``resume_experiment_id``: if given, reuse that experiment id instead of
    minting a new one, and skip (model, temperature_condition, prompt_variant,
    run_index) combinations that already have a valid record on disk for it --
    lets an interrupted run be continued with `main.py run --resume <id>`
    instead of restarting (and re-billing) everything.
    """
    out_dir = Path(settings.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    experiment_id = resume_experiment_id or new_experiment_id()

    active: list[ModelConfig] = []
    for m in models:
        if not m.enabled:
            continue
        if only and not _matches_only(m.name, only):
            continue
        adapter = build_adapter(m.provider, m.model_id, m.params)
        if not adapter.is_available():
            log(f"[skip] {m.name}: {adapter.api_key_env} not set -- skipping.")
            continue
        active.append(m)

    if only:
        matched_names = {m.name for m in active}
        for token in only:
            if not any(_matches_only(name, [token]) for name in matched_names) and not any(
                _matches_only(m.name, [token]) for m in models
            ):
                log(f"[warn] --only token '{token}' matched no configured model.")

    if not active:
        log("No runnable models (check API keys and config/models.yaml).")
        return []

    already_done = _existing_valid_keys(out_dir, experiment_id) if resume_experiment_id else set()

    jobs = [
        (m, tc, pv, i)
        for m in active
        for tc in settings.temperature_conditions
        for pv in settings.prompt_variants
        for i in range(settings.n_runs)
    ]
    skipped = [j for j in jobs if (j[0].name, j[1].label, j[2], j[3]) in already_done]
    jobs = [j for j in jobs if (j[0].name, j[1].label, j[2], j[3]) not in already_done]

    log(
        f"Experiment {experiment_id}: running {len(active)} model(s) x "
        f"{len(settings.temperature_conditions)} temperature condition(s) x "
        f"{len(settings.prompt_variants)} prompt variant(s) x {settings.n_runs} run(s) = "
        f"{len(jobs)} calls"
        + (f" ({len(skipped)} already valid, resuming '{resume_experiment_id}')." if skipped else ".")
    )

    records: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, settings.max_concurrency)) as pool:
        futures = {
            pool.submit(_safe_run, m, inst, settings, i, tc, pv): (m.name, tc.label, pv, i)
            for m, tc, pv, i in jobs
        }
        for fut in as_completed(futures):
            name, cond_label, pv, idx = futures[fut]
            record = fut.result()
            record["experiment_id"] = experiment_id
            path = _write_record(record, out_dir)
            records.append(record)
            status = "ok" if record["valid"] else "INVALID"
            type_str = record.get("score", {}).get("type", "----")
            log(
                f"[{status}] {name} [{cond_label}/{pv}] run {idx:02d} -> {type_str}  "
                f"({path.name})"
            )

    valid = sum(1 for r in records if r["valid"])
    log(f"Done. {valid}/{len(records)} valid runs written to {out_dir} (experiment {experiment_id}).")
    return records


def _matches_only(model_name: str, only: list[str]) -> bool:
    """`--only` matches by exact name or by prefix (e.g. `gpt` matches every
    gpt-5 tier), so a same-family shorthand selects the whole family."""
    return any(model_name == tok or model_name.startswith(tok) for tok in only)


def _existing_valid_keys(out_dir: Path, experiment_id: str) -> set[tuple[str, str, str, int]]:
    """(model_name, temperature_condition, prompt_variant, run_index) already
    written as a *valid* record under this experiment id -- used by --resume
    to avoid re-running (and re-billing) completed work."""
    keys: set[tuple[str, str, str, int]] = set()
    for path in out_dir.glob("*.json"):
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if rec.get("experiment_id") == experiment_id and rec.get("valid"):
            keys.add(
                (
                    rec.get("model_name"),
                    rec.get("temperature_condition", "default"),
                    rec.get("prompt_variant", "default"),
                    rec.get("run_index"),
                )
            )
    return keys


def _safe_run(
    model: ModelConfig,
    inst: Instrument,
    settings: RunSettings,
    i: int,
    temperature_condition: TemperatureCondition,
    prompt_variant: str,
) -> dict:
    """Wrapper so a single crashing run never kills the whole batch."""
    try:
        return execute_single_run(model, inst, settings, i, temperature_condition, prompt_variant)
    except Exception as exc:  # noqa: BLE001
        return {
            "schema_version": SCHEMA_VERSION,
            "timestamp": _now_iso(),
            "model_name": model.name,
            "provider": model.provider,
            "model_id": model.model_id,
            "run_index": i,
            "temperature_condition": temperature_condition.label,
            "temperature_value": temperature_condition.value,
            "prompt_variant": prompt_variant,
            "valid": False,
            "succeeded_at_attempt": None,
            "invalid_reason": f"unexpected: {exc}",
            "traceback": traceback.format_exc(),
            "attempts": [],
        }
