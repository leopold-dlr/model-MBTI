#!/usr/bin/env python3
"""LLM MBTI Arena — CLI entry point.

Examples:
  # Full pipeline: run every configured model, then build reports.
  python main.py run

  # Only re-generate reports from existing data/runs (no API calls).
  python main.py report

  # Run a subset of models (exact name or family prefix, e.g. "gpt" matches
  # gpt-5/gpt-5-mini/gpt-5-nano).
  python main.py run --only claude-sonnet gpt

  # Resume an interrupted run instead of re-billing everything.
  python main.py run --resume 20260719T120000Z

  # Build reports from one specific experiment instead of the latest.
  python main.py report --experiment 20260719T120000Z
  python main.py report --list-experiments

  # Preview the exact prompt that will be sent (no API calls).
  python main.py show-prompt
  python main.py show-prompt --model gpt-5 --run-index 3 --condition fixed_t1

The user only needs to supply API keys in `.env`.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

from src.config import load_models, load_run_settings
from src.instrument import load_instrument
from src.prompting import templates
from src.report.aggregate import aggregate_all, list_experiment_ids, load_records
from src.report.render import write_reports
from src.runner.orchestrator import _run_seed, run_experiment

ROOT = Path(__file__).resolve().parent
DEFAULT_MODELS = ROOT / "config" / "models.yaml"
DEFAULT_SETTINGS = ROOT / "config" / "run_settings.yaml"
DEFAULT_INSTRUMENT = ROOT / "config" / "instrument" / "oejts_32.yaml"


def _most_common_n_total(stats) -> int | None:
    """The N the baseline simulation should use: the most common n_total
    across the loaded (model, condition) groups, since that's how many runs
    this experiment actually collected -- not whatever run_settings.yaml
    happens to say n_runs is right now."""
    totals = [s.n_total for s in stats if s.n_total]
    return Counter(totals).most_common(1)[0][0] if totals else None


def _load_all(args):
    inst = load_instrument(args.instrument)
    settings = load_run_settings(args.settings)
    models = load_models(args.models)
    return inst, settings, models


def cmd_run(args) -> int:
    inst, settings, models = _load_all(args)
    records = run_experiment(
        models, inst, settings, only=args.only or None, resume_experiment_id=args.resume
    )
    if not args.no_report:
        _build_reports(settings, inst, experiment_id=args.resume)
    return 0 if records else 1


def cmd_report(args) -> int:
    inst, settings, _ = _load_all(args)
    if args.list_experiments:
        records = load_records(settings.output_dir)
        ids = list_experiment_ids(records)
        if not ids:
            print(f"No experiment ids found in {settings.output_dir}.")
        else:
            print("Experiments found (oldest first):")
            for eid in ids:
                marker = " (latest)" if eid == ids[-1] else ""
                print(f"  {eid}{marker}")
        return 0
    experiment_id = None if args.all_experiments else args.experiment
    _build_reports(settings, inst, experiment_id=experiment_id)
    return 0


def _build_reports(settings, inst, experiment_id=None):
    records = load_records(settings.output_dir)
    if not records:
        print(f"No run records found in {settings.output_dir}. Run `python main.py run` first.")
        return
    stats = aggregate_all(
        records, inst=inst, experiment_id=experiment_id, min_valid_runs=settings.min_valid_runs
    )
    used_ids = {r.get("experiment_id") for r in records if r.get("experiment_id")}
    if experiment_id is None and used_ids:
        experiment_id = max(used_ids)
    print(f"Building reports for experiment: {experiment_id or '(no experiment_id in data)'}")
    # Baseline must reflect how many runs were actually loaded, not the live
    # config's current n_runs -- those silently diverge whenever run_settings
    # is edited (e.g. lowered for a smoke test, restored afterward) and
    # `report` is re-run against the older data without a fresh `run`.
    n_runs_seen = _most_common_n_total(stats)
    paths = write_reports(stats, inst.type_order, settings.report_dir, inst=inst, n_runs=n_runs_seen)
    print("Reports written:")
    for kind, p in paths.items():
        print(f"  {kind:10s} {p}")


def cmd_show_prompt(args) -> int:
    inst, settings, models = _load_all(args)
    if args.model:
        run_index = args.run_index
        seed = _run_seed(settings.seed, args.model, run_index)
        note = f" (model={args.model}, run_index={run_index}, seed={seed})"
    else:
        seed = settings.seed
        note = " (generic preview -- not tied to any specific model/run; pass --model to reproduce an exact run)"
    flips = templates.flip_map(inst, seed)
    system, user, order = templates.build_prompt(
        inst,
        seed=seed,
        randomize=settings.randomize_item_order,
        flips=flips,
        prompt_variant=args.prompt_variant,
    )
    print("=== SYSTEM PROMPT ===" + note + "\n" + system)
    print("\n=== USER PROMPT ===\n" + user)
    print("\n=== ITEM ORDER ===\n" + ", ".join(map(str, order)))
    print("\n=== FLIPPED ITEM IDS (polarity counterbalance) ===\n" + ", ".join(map(str, sorted(flips))))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Administer the OEJTS to LLMs and compare types.")
    p.add_argument("--models", default=str(DEFAULT_MODELS))
    p.add_argument("--settings", default=str(DEFAULT_SETTINGS))
    p.add_argument("--instrument", default=str(DEFAULT_INSTRUMENT))
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="Run models then build reports.")
    r.add_argument(
        "--only", nargs="*", help="Restrict to these model names (exact match or family prefix)."
    )
    r.add_argument("--no-report", action="store_true", help="Skip report generation.")
    r.add_argument(
        "--resume",
        default=None,
        help="Experiment id to resume (skips model/condition/run combos already valid on disk "
        "for it) instead of starting a new experiment and re-billing everything.",
    )
    r.set_defaults(func=cmd_run)

    rep = sub.add_parser("report", help="Rebuild reports from existing runs (no API calls).")
    rep.add_argument(
        "--experiment", default=None, help="Build reports from this experiment id only (default: latest)."
    )
    rep.add_argument(
        "--all-experiments",
        action="store_true",
        help="Pool every experiment id together (legacy behavior; not recommended).",
    )
    rep.add_argument(
        "--list-experiments", action="store_true", help="List experiment ids found in output_dir and exit."
    )
    rep.set_defaults(func=cmd_report)

    sp = sub.add_parser("show-prompt", help="Print the prompt without calling any API.")
    sp.add_argument("--model", default=None, help="Model name, to reproduce its exact per-run seed.")
    sp.add_argument("--run-index", type=int, default=0)
    sp.add_argument("--prompt-variant", default="default")
    sp.set_defaults(func=cmd_show_prompt)
    return p


def main(argv=None) -> int:
    if load_dotenv is not None:
        load_dotenv(ROOT / ".env")
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
