"""Smoke tests for aggregation + report rendering on synthetic run records."""

from pathlib import Path

from src.instrument import load_instrument
from src.report.aggregate import aggregate_all
from src.report.render import render_dashboard, render_markdown, render_paper
from src.scoring.mbti_scorer import score_answers

INSTRUMENT = Path(__file__).resolve().parent.parent / "config" / "instrument" / "oejts_32.yaml"


def _make_record(name, answers, valid=True, run_index=0):
    inst = load_instrument(INSTRUMENT)
    rec = {
        "model_name": name,
        "provider": "test",
        "model_id": f"{name}-id",
        "run_index": run_index,
        "valid": valid,
    }
    if valid:
        rec["score"] = score_answers(inst, answers).to_dict()
    else:
        rec["invalid_reason"] = "parse_error: refusal"
    return rec


def _synthetic_records():
    inst = load_instrument(INSTRUMENT)
    ext = {it.id: 5 for it in inst.items}  # -> INTP
    recs = []
    # A perfectly stable model.
    for i in range(5):
        recs.append(_make_record("stable", ext, run_index=i))
    # A model that flips one axis half the time + one invalid run.
    for i in range(4):
        ans = dict(ext)
        if i % 2 == 0:
            for it in inst.items_for_axis("TF"):
                ans[it.id] = 1  # flip T->F
        recs.append(_make_record("wobbly", ans, run_index=i))
    recs.append(_make_record("wobbly", None, valid=False, run_index=4))
    return recs


def test_aggregate_stability():
    stats = {s.model_name: s for s in aggregate_all(_synthetic_records())}
    stable = stats["stable"]
    assert stable.n_valid == 5
    assert stable.modal_type == "INTP"
    assert stable.modal_type_freq == 1.0
    assert stable.axes["TF"].modal_freq == 1.0

    wobbly = stats["wobbly"]
    assert wobbly.n_valid == 4
    assert wobbly.n_invalid == 1
    # TF flips 2/4 -> modal freq 0.5, dispersion > 0
    assert wobbly.axes["TF"].modal_freq == 0.5
    assert wobbly.axes["TF"].std_pct_high > 0


def test_renders_do_not_crash():
    inst = load_instrument(INSTRUMENT)
    stats = aggregate_all(_synthetic_records())
    md = render_markdown(stats, inst.type_order)
    paper = render_paper(stats, inst.type_order)
    html = render_dashboard(stats, inst.type_order)
    assert "Comparative Report" in md
    assert "OEJTS" in paper
    assert "<!doctype html>" in html
    assert "__DATA_JSON__" not in html  # placeholder was substituted
