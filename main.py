#!/usr/bin/env python3
"""LLM MBTI Arena — CLI entry point.

Examples:
  # Full pipeline: run every configured model, then build reports.
  python main.py run

  # Only re-generate reports from existing data/runs (no API calls).
  python main.py report

  # Run a subset of models.
  python main.py run --only claude-sonnet gpt

  # Preview the exact prompt that will be sent (no API calls).
  python main.py show-prompt

The user only needs to supply API keys in `.env`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

from src.config import load_models, load_run_settings
from src.instrument import load_instrument
from src.prompting import templates
from src.report.aggregate import aggregate_all, load_records
from src.report.render import write_reports
from src.runner.orchestrator import run_experiment

ROOT = Path(__file__).resolve().parent
DEFAULT_MODELS = ROOT / "config" / "models.yaml"
DEFAULT_SETTINGS = ROOT / "config" / "run_settings.yaml"
DEFAULT_INSTRUMENT = ROOT / "config" / "instrument" / "oejts_32.yaml"


def _load_all(args):
    inst = load_instrument(args.instrument)
    settings = load_run_settings(args.settings)
    models = load_models(args.models)
    return inst, settings, models


def cmd_run(args) -> int:
    inst, settings, models = _load_all(args)
    records = run_experiment(models, inst, settings, only=args.only or None)
    if not args.no_report:
        _build_reports(settings, inst)
    return 0 if records else 1


def cmd_report(args) -> int:
    inst, settings, _ = _load_all(args)
    _build_reports(settings, inst)
    return 0


def _build_reports(settings, inst):
    records = load_records(settings.output_dir)
    if not records:
        print(f"No run records found in {settings.output_dir}. Run `python main.py run` first.")
        return
    stats = aggregate_all(records)
    paths = write_reports(stats, inst.type_order, settings.report_dir)
    print("Reports written:")
    for kind, p in paths.items():
        print(f"  {kind:10s} {p}")


def cmd_show_prompt(args) -> int:
    inst, settings, _ = _load_all(args)
    system, user, order = templates.build_prompt(
        inst, seed=settings.seed, randomize=settings.randomize_item_order
    )
    print("=== SYSTEM PROMPT ===\n" + system)
    print("\n=== USER PROMPT ===\n" + user)
    print("\n=== ITEM ORDER ===\n" + ", ".join(map(str, order)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Administer the OEJTS to LLMs and compare types.")
    p.add_argument("--models", default=str(DEFAULT_MODELS))
    p.add_argument("--settings", default=str(DEFAULT_SETTINGS))
    p.add_argument("--instrument", default=str(DEFAULT_INSTRUMENT))
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="Run models then build reports.")
    r.add_argument("--only", nargs="*", help="Restrict to these model names.")
    r.add_argument("--no-report", action="store_true", help="Skip report generation.")
    r.set_defaults(func=cmd_run)

    rep = sub.add_parser("report", help="Rebuild reports from existing runs (no API calls).")
    rep.set_defaults(func=cmd_report)

    sp = sub.add_parser("show-prompt", help="Print the prompt without calling any API.")
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
