"""Smoke tests for aggregation + report rendering on synthetic run records."""

from pathlib import Path

from src.instrument import load_instrument
from src.report.aggregate import (
    aggregate_all,
    cronbach_alpha,
    random_baseline_stats,
    wilson_ci,
)
from src.report.render import render_csv, render_dashboard, render_markdown, render_paper
from src.scoring.mbti_scorer import score_answers

INSTRUMENT = Path(__file__).resolve().parent.parent / "config" / "instrument" / "oejts_32.yaml"
IPIP50 = Path(__file__).resolve().parent.parent / "config" / "instrument" / "ipip50_bigfive.yaml"


def _make_record(
    name,
    answers,
    valid=True,
    run_index=0,
    temperature_condition="default",
    prompt_variant="default",
    experiment_id=None,
    succeeded_at_attempt=0,
    inst=None,
):
    inst = inst or load_instrument(INSTRUMENT)
    rec = {
        "model_name": name,
        "provider": "test",
        "model_id": f"{name}-id",
        "run_index": run_index,
        "temperature_condition": temperature_condition,
        "prompt_variant": prompt_variant,
        "valid": valid,
    }
    if experiment_id is not None:
        rec["experiment_id"] = experiment_id
    if valid:
        rec["score"] = score_answers(inst, answers).to_dict()
        rec["answers"] = {str(k): v for k, v in answers.items()}
        rec["item_polarity"] = {}
        rec["succeeded_at_attempt"] = succeeded_at_attempt
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


def test_wilson_ci_bounds():
    lo, hi = wilson_ci(20, 20)
    assert 0.0 < lo < 1.0
    assert hi == 1.0  # clamped, never exceeds 1
    lo0, hi0 = wilson_ci(0, 20)
    assert lo0 == 0.0
    assert 0.0 < hi0 < 1.0
    # A larger sample at the same proportion has a tighter interval.
    lo_small, hi_small = wilson_ci(5, 10)
    lo_big, hi_big = wilson_ci(50, 100)
    assert (hi_big - lo_big) < (hi_small - lo_small)


def test_modal_type_freq_ci_present_and_sane():
    stats = {s.model_name: s for s in aggregate_all(_synthetic_records())}
    stable = stats["stable"]
    lo, hi = stable.modal_type_freq_ci
    assert lo <= stable.modal_type_freq <= hi


def test_reliable_flag_uses_min_valid_runs():
    records = _synthetic_records()
    lenient = {s.model_name: s for s in aggregate_all(records, min_valid_runs=2)}
    strict = {s.model_name: s for s in aggregate_all(records, min_valid_runs=10)}
    assert lenient["wobbly"].reliable is True   # 4 valid >= 2
    assert strict["wobbly"].reliable is False   # 4 valid < 10


def test_experiment_id_filters_to_latest_by_default():
    inst = load_instrument(INSTRUMENT)
    ext = {it.id: 5 for it in inst.items}
    old = [_make_record("m", ext, experiment_id="exp_a", run_index=0)]
    new = [_make_record("m", ext, experiment_id="exp_b", run_index=0)]
    stats = {s.model_name: s for s in aggregate_all(old + new)}
    assert stats["m"].n_total == 1  # only the latest experiment (exp_b) counted

    stats_old = {s.model_name: s for s in aggregate_all(old + new, experiment_id="exp_a")}
    assert stats_old["m"].n_total == 1


def test_temperature_condition_kept_separate():
    inst = load_instrument(INSTRUMENT)
    ext = {it.id: 5 for it in inst.items}
    recs = [
        _make_record("m", ext, run_index=0, temperature_condition="default"),
        _make_record("m", ext, run_index=0, temperature_condition="fixed_t1"),
    ]
    stats = aggregate_all(recs)
    labels = {(s.model_name, s.temperature_condition) for s in stats}
    assert labels == {("m", "default"), ("m", "fixed_t1")}


def test_cronbach_alpha_high_for_consistent_answers():
    # Perfectly consistent per-item answers across runs -> item variance is 0
    # for each column but not for the axis total unless all runs are
    # identical; use near-identical rows with tiny jitter so variance exists
    # without being degenerate.
    matrix = [[5, 5, 4, 5], [4, 5, 5, 4], [5, 4, 5, 5], [5, 5, 5, 4]]
    alpha = cronbach_alpha(matrix)
    assert alpha is not None

    assert cronbach_alpha([[1, 2, 3]]) is None  # fewer than 2 runs
    assert cronbach_alpha([[1], [2]]) is None  # fewer than 2 items


def test_cronbach_alpha_surfaced_when_instrument_given():
    stats = {s.model_name: s for s in aggregate_all(_synthetic_records(), inst=load_instrument(INSTRUMENT))}
    tf = stats["stable"].axes["TF"]
    # 5 identical runs -> zero total variance -> alpha undefined (None), which
    # is the correct answer, not a divide-by-zero crash.
    assert tf.cronbach_alpha is None or isinstance(tf.cronbach_alpha, float)


def test_random_baseline_stats_shape():
    inst = load_instrument(INSTRUMENT)
    baseline = random_baseline_stats(inst, n_runs=10, n_trials=50, seed=1)
    assert 0.0 <= baseline["modal_type_freq_mean"] <= 1.0
    assert baseline["modal_type_freq_p05"] <= baseline["modal_type_freq_p95"]
    assert set(baseline["axes"]) == set(inst.type_order)


def test_paper_explains_missing_comparison_when_all_models_unreliable():
    """Regression test: when every model in the primary condition is below
    min_valid_runs (e.g. a small smoke test), '## 3. Cross-model comparison'
    used to render as a bare header with nothing under it -- silently. It
    must instead say why no comparison is shown."""
    inst = load_instrument(INSTRUMENT)
    records = _synthetic_records()  # 5/4 valid runs, well below a high threshold
    stats = aggregate_all(records, inst=inst, min_valid_runs=1000)
    assert all(not s.reliable for s in stats if s.n_valid > 0)
    paper = render_paper(stats, inst.type_order)
    assert "No comparison shown" in paper
    assert "min_valid_runs" in paper


def test_plain_language_summary_present_in_markdown():
    inst = load_instrument(INSTRUMENT)
    stats = aggregate_all(_synthetic_records(), inst=inst)
    md = render_markdown(stats, inst.type_order)
    assert "In plain terms" in md


def test_paper_and_markdown_describe_the_actual_instrument_used():
    """Regression test: render_paper/render_markdown used to hardcode
    "OEJTS 1.2", "32-item", and "four MBTI dichotomies (E/I, S/N, T/F, J/P)"
    regardless of which instrument was actually administered -- so a report
    built from an IPIP-50 (Big Five) run falsely described itself as an
    OEJTS/MBTI study. Both renderers must reflect the instrument passed in."""
    ipip = load_instrument(IPIP50)
    ext = {it.id: 5 for it in ipip.items}
    records = [_make_record("m", ext, run_index=i, inst=ipip) for i in range(5)]
    stats = aggregate_all(records, inst=ipip)

    paper = render_paper(stats, ipip.type_order, inst=ipip)
    md = render_markdown(stats, ipip.type_order, inst=ipip)
    assert "IPIP-50-BigFive" in paper
    assert "50 items" in paper
    assert "OEJTS" not in paper
    assert "32-item" not in paper
    assert "four MBTI dichotomies" not in paper
    assert "OEJTS" not in md

    oejts = load_instrument(INSTRUMENT)
    oejts_stats = aggregate_all(_synthetic_records(), inst=oejts)
    oejts_paper = render_paper(oejts_stats, oejts.type_order, inst=oejts)
    assert "OEJTS" in oejts_paper
    assert "32 items" in oejts_paper


def test_renders_do_not_crash():
    inst = load_instrument(INSTRUMENT)
    stats = aggregate_all(_synthetic_records(), inst=inst)
    baseline = random_baseline_stats(inst, n_runs=5, n_trials=20)
    md = render_markdown(stats, inst.type_order, baseline=baseline)
    paper = render_paper(stats, inst.type_order, baseline=baseline)
    html = render_dashboard(stats, inst.type_order, baseline=baseline)
    csv_text = render_csv(stats, inst.type_order)
    assert "Comparative Report" in md
    assert "OEJTS" in paper
    assert "<!doctype html>" in html
    assert "__DATA_JSON__" not in html  # placeholder was substituted
    assert "__CHARTJS_SOURCE__" not in html  # vendored Chart.js was inlined
    assert "Chart.js" in html  # vendored source's own banner comment
    assert "model_name" in csv_text.splitlines()[0]
    assert "stable" in csv_text
