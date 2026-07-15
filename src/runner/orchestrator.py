"""Drive the whole experiment: for each model x N runs, prompt -> parse ->
score -> persist a raw run file.

Every raw run is stored in full (prompt sent, raw reply, parsed answers, score,
usage) so the scoring can be replayed later without re-hitting the APIs, and so
refusals/invalid runs remain auditable.
"""

from __future__ import annotations

import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from ..config import ModelConfig, RunSettings
from ..instrument import Instrument
from ..prompting import templates
from ..prompting.parser import ParseError, parse_answers
from ..providers.base import ProviderError
from ..providers.registry import build_adapter
from ..scoring.mbti_scorer import score_answers


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_seed(base_seed: int, model_name: str, run_index: int) -> int:
    """Deterministic per-run seed so item order is reproducible."""
    h = abs(hash((base_seed, model_name, run_index)))
    return (base_seed + run_index * 1000 + (h % 100000)) & 0x7FFFFFFF


def execute_single_run(
    model: ModelConfig,
    inst: Instrument,
    settings: RunSettings,
    run_index: int,
) -> dict:
    """Execute one run (with retries). Returns a fully-populated run record."""
    adapter = build_adapter(model.provider, model.model_id, model.params)
    seed = _run_seed(settings.seed, model.name, run_index)
    randomize = settings.randomize_item_order

    record: dict = {
        "schema_version": 1,
        "timestamp": _now_iso(),
        "model_name": model.name,
        "provider": model.provider,
        "model_id": model.model_id,
        "instrument": {"name": inst.name, "version": inst.version},
        "run_index": run_index,
        "seed": seed,
        "language": settings.language,
        "mode": settings.mode,
        "valid": False,
        "attempts": [],
    }

    last_error = None
    for attempt in range(settings.max_retries + 1):
        is_retry = attempt > 0
        system, user, id_order = templates.build_prompt(
            inst, seed=seed, randomize=randomize, retry=is_retry
        )
        attempt_log: dict = {"attempt": attempt, "is_retry": is_retry}
        try:
            gen = adapter.generate(system, user)
        except ProviderError as exc:
            attempt_log["error"] = f"provider_error: {exc}"
            record["attempts"].append(attempt_log)
            last_error = str(exc)
            time.sleep(min(2 ** attempt, 8))
            continue

        attempt_log["raw_response"] = gen.text
        attempt_log["usage"] = gen.usage
        attempt_log["returned_model"] = gen.model

        try:
            answers = parse_answers(
                gen.text, id_order, inst.scale_min, inst.scale_max
            )
        except ParseError as exc:
            attempt_log["error"] = f"parse_error: {exc}"
            record["attempts"].append(attempt_log)
            last_error = str(exc)
            continue

        # Success: score it.
        result = score_answers(inst, answers)
        attempt_log["answers"] = {str(k): v for k, v in answers.items()}
        record["attempts"].append(attempt_log)
        record["valid"] = True
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
    fname = f"{record['model_name']}__run{record['run_index']:02d}__{ts}.json"
    path = out_dir / fname
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def run_experiment(
    models: list[ModelConfig],
    inst: Instrument,
    settings: RunSettings,
    only: list[str] | None = None,
    log=print,
) -> list[dict]:
    """Run every enabled (and available) model x n_runs. Returns all records."""
    out_dir = Path(settings.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    active: list[ModelConfig] = []
    for m in models:
        if not m.enabled:
            continue
        if only and m.name not in only:
            continue
        adapter = build_adapter(m.provider, m.model_id, m.params)
        if not adapter.is_available():
            log(f"[skip] {m.name}: {adapter.api_key_env} not set -- skipping.")
            continue
        active.append(m)

    if not active:
        log("No runnable models (check API keys and config/models.yaml).")
        return []

    log(
        f"Running {len(active)} model(s) x {settings.n_runs} run(s) = "
        f"{len(active) * settings.n_runs} calls."
    )

    jobs = [(m, i) for m in active for i in range(settings.n_runs)]
    records: list[dict] = []

    with ThreadPoolExecutor(max_workers=max(1, settings.max_concurrency)) as pool:
        futures = {
            pool.submit(_safe_run, m, inst, settings, i): (m.name, i) for m, i in jobs
        }
        for fut in as_completed(futures):
            name, idx = futures[fut]
            record = fut.result()
            path = _write_record(record, out_dir)
            records.append(record)
            status = "ok" if record["valid"] else "INVALID"
            type_str = record.get("score", {}).get("type", "----")
            log(f"[{status}] {name} run {idx:02d} -> {type_str}  ({path.name})")

    valid = sum(1 for r in records if r["valid"])
    log(f"Done. {valid}/{len(records)} valid runs written to {out_dir}.")
    return records


def _safe_run(model: ModelConfig, inst: Instrument, settings: RunSettings, i: int) -> dict:
    """Wrapper so a single crashing run never kills the whole batch."""
    try:
        return execute_single_run(model, inst, settings, i)
    except Exception as exc:  # noqa: BLE001
        return {
            "schema_version": 1,
            "timestamp": _now_iso(),
            "model_name": model.name,
            "provider": model.provider,
            "model_id": model.model_id,
            "run_index": i,
            "valid": False,
            "invalid_reason": f"unexpected: {exc}",
            "traceback": traceback.format_exc(),
            "attempts": [],
        }
