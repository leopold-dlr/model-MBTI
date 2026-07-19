"""Render aggregated stats into (1) a comparative markdown report, (2) an
interactive self-contained HTML dashboard, (3) a CSV export, and (4) a
synthesis paper skeleton.

All four are generated from the same aggregated data, so re-running the
pipeline refreshes them without any manual editing.
"""

from __future__ import annotations

import csv
import io
import json
from collections import Counter
from datetime import date
from pathlib import Path

from ..instrument import Instrument
from .aggregate import ModelStats, random_baseline_stats

# 20-color categorical palette (matplotlib's "tab20"), so up to 20 distinct
# model series stay visually distinguishable -- the previous 10-color palette
# repeated colors past model #10, which for a 20-model portfolio makes half
# the series indistinguishable in the comparison charts.
PALETTE = [
    "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
    "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
    "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",
    "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5",
]

_VENDOR_DIR = Path(__file__).resolve().parent / "vendor"


def condition_label(s: ModelStats) -> str:
    return f"{s.temperature_condition}/{s.prompt_variant}"


def _group_by_condition(stats: list[ModelStats]) -> list[tuple[str, list[ModelStats]]]:
    """Group stats by (temperature_condition, prompt_variant), preserving the
    order conditions first appear in. Most reports will have exactly one
    group (single condition/variant); multiple groups only appear once you
    opt into the temperature or prompt-variant ablation in run_settings.yaml."""
    order: list[str] = []
    groups: dict[str, list[ModelStats]] = {}
    for s in stats:
        label = condition_label(s)
        if label not in groups:
            groups[label] = []
            order.append(label)
        groups[label].append(s)
    return [(label, groups[label]) for label in order]


def _fmt_ci(ci: tuple[float, float] | list[float]) -> str:
    lo, hi = ci
    return f"[{lo*100:.0f},{hi*100:.0f}]"


# --------------------------------------------------------------------------- #
# Markdown comparative report
# --------------------------------------------------------------------------- #
def render_markdown(
    stats: list[ModelStats],
    axes_order: list[str],
    baseline: dict | None = None,
) -> str:
    groups = _group_by_condition(stats)
    lines = [
        "# LLM MBTI Arena — Comparative Report",
        "",
        f"_Generated {date.today().isoformat()} from {sum(s.n_total for s in stats)} "
        f"runs across {len({s.model_name for s in stats})} models, "
        f"{len(groups)} condition(s)._",
        "",
        "Stability = share of valid runs landing on the model's modal type/letter, "
        "with a 95% Wilson confidence interval in brackets -- at N<=20 runs, do not "
        "read small differences between models as meaningful without checking "
        "whether their intervals overlap. Per-axis cells show only the modal "
        "letter and its run frequency; mean preference-% and Cronbach's alpha "
        "(inter-item consistency, None if undefined) are in the per-model detail "
        "below, not in this summary table, to avoid cramming two different "
        "percentages into one cell.",
        "",
    ]
    if baseline is not None:
        lines += [
            f"**Random-responder baseline** (Monte Carlo, {baseline['n_trials']} simulated "
            f"models of {baseline['n_runs']} runs each, uniform random answers, zero signal): "
            f"expected modal-type stability = {baseline['modal_type_freq_mean']*100:.0f}% "
            f"(5th-95th pct: {baseline['modal_type_freq_p05']*100:.0f}-"
            f"{baseline['modal_type_freq_p95']*100:.0f}%). A model's stability is only "
            "evidence of a real response tendency to the extent it clears this floor.",
            "",
        ]

    for label, group in groups:
        if len(groups) > 1:
            lines += [f"## Condition: `{label}`", ""]
            tc = group[0].temperature_condition
            if tc == "default":
                lines.append(
                    "_Provider default temperature -- confounded by differing sampling "
                    "entropy across providers; do not use this condition alone to compare "
                    "stability ACROSS models._"
                )
            else:
                lines.append(
                    "_Fixed temperature across all models -- the controlled condition for "
                    "cross-model stability comparison._"
                )
            lines.append("")

        header = "| Model | Provider | Modal type | Stability (95% CI) | Reliable | Valid |"
        sep = "|---|---|---|---|---|---|"
        for axis in axes_order:
            header += f" {axis} |"
            sep += "---|"
        lines += [header, sep]

        for s in group:
            reliable_mark = "✓" if s.reliable else "⚠ low N"
            row = (
                f"| `{s.model_name}` | {s.provider} | **{s.modal_type}** "
                f"| {s.modal_type_freq*100:.0f}% {_fmt_ci(s.modal_type_freq_ci)} "
                f"| {reliable_mark} | {s.n_valid}/{s.n_total} |"
            )
            for axis in axes_order:
                a = s.axes.get(axis)
                if a is None:
                    row += " — |"
                    continue
                row += f" {a.modal_letter} ({a.modal_freq*100:.0f}%) |"
            lines.append(row)
        lines.append("")

    lines += ["## Per-model detail", ""]
    for label, group in groups:
        cond_suffix = f" — condition `{label}`" if len(groups) > 1 else ""
        for s in group:
            lines += [f"### `{s.model_name}`{cond_suffix} — {s.provider} / `{s.model_id}`", ""]
            if s.n_valid == 0:
                lines += [
                    f"- No valid runs ({s.n_invalid} invalid).",
                    _invalid_summary(s),
                    "",
                ]
                continue
            reliability_note = (
                "" if s.reliable else f" — **below min_valid_runs threshold, treat with caution**"
            )
            lines += [
                f"- Modal type: **{s.modal_type}** (stable in {s.modal_type_freq*100:.0f}% "
                f"of runs, 95% CI {_fmt_ci(s.modal_type_freq_ci)}){reliability_note}",
                f"- Valid runs: {s.n_valid}/{s.n_total}"
                + (f" ({s.n_invalid} invalid)" if s.n_invalid else ""),
                f"- Answered without any retry: {s.pct_first_attempt*100:.0f}% of valid runs",
                f"- Types observed: {_fmt_counts(s.type_counts)}",
            ]
            for axis in axes_order:
                a = s.axes.get(axis)
                if a is None:
                    continue
                alpha_str = f"{a.cronbach_alpha:.2f}" if a.cronbach_alpha is not None else "n/a"
                lines.append(
                    f"- {axis}: modal **{a.modal_letter}** "
                    f"({a.modal_freq*100:.0f}% of runs, CI {_fmt_ci(a.modal_freq_ci)}); "
                    f"mean pref toward {a.pole_high}={a.mean_pct_high:.0f}% "
                    f"(σ={a.std_pct_high:.1f}, distance from midpoint={a.dist_from_midpoint:.0f}pt); "
                    f"Cronbach α={alpha_str}; letters {_fmt_counts(a.letter_counts)}"
                )
            if s.n_invalid:
                lines.append(_invalid_summary(s))
            lines.append("")

    lines += [
        "## Methodological caveats",
        "",
        "See the project README. In short: the OEJTS is an open MBTI-style "
        "instrument whose validity is contested even for humans. Applied to an "
        "LLM it measures a *prompt-conditioned text-output tendency*, not a "
        "personality trait. Treat this as exploratory, not psychometric. A low "
        "or undefined Cronbach's alpha on an axis means that axis's items don't "
        "covary for that model -- its letter is not measuring one coherent "
        "thing, whatever its apparent run-to-run stability.",
        "",
    ]
    return "\n".join(lines)


def _fmt_counts(counts: dict) -> str:
    if not counts:
        return "—"
    return ", ".join(f"{k}×{v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))


def _invalid_summary(s: ModelStats) -> str:
    if not s.invalid_reasons:
        return ""
    c = Counter(r[:80] for r in s.invalid_reasons)
    reasons = "; ".join(f"{k} (×{v})" for k, v in c.most_common(3))
    return f"- Invalid runs: {reasons}"


# --------------------------------------------------------------------------- #
# CSV export
# --------------------------------------------------------------------------- #
def render_csv(stats: list[ModelStats], axes_order: list[str]) -> str:
    """One row per (model, temperature_condition, prompt_variant). Axis
    columns are prefixed per axis so the file stays flat/spreadsheet-friendly."""
    fieldnames = [
        "model_name", "provider", "model_id", "temperature_condition", "prompt_variant",
        "n_total", "n_valid", "n_invalid", "reliable", "pct_first_attempt",
        "modal_type", "modal_type_freq", "modal_type_freq_ci_lo", "modal_type_freq_ci_hi",
    ]
    for axis in axes_order:
        fieldnames += [
            f"{axis}_modal_letter", f"{axis}_modal_freq",
            f"{axis}_modal_freq_ci_lo", f"{axis}_modal_freq_ci_hi",
            f"{axis}_mean_pct_high", f"{axis}_std_pct_high",
            f"{axis}_dist_from_midpoint", f"{axis}_cronbach_alpha",
        ]

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for s in stats:
        row = {
            "model_name": s.model_name,
            "provider": s.provider,
            "model_id": s.model_id,
            "temperature_condition": s.temperature_condition,
            "prompt_variant": s.prompt_variant,
            "n_total": s.n_total,
            "n_valid": s.n_valid,
            "n_invalid": s.n_invalid,
            "reliable": s.reliable,
            "pct_first_attempt": round(s.pct_first_attempt, 4),
            "modal_type": s.modal_type,
            "modal_type_freq": round(s.modal_type_freq, 4),
            "modal_type_freq_ci_lo": round(s.modal_type_freq_ci[0], 4),
            "modal_type_freq_ci_hi": round(s.modal_type_freq_ci[1], 4),
        }
        for axis in axes_order:
            a = s.axes.get(axis)
            if a is None:
                for suffix in (
                    "modal_letter", "modal_freq", "modal_freq_ci_lo", "modal_freq_ci_hi",
                    "mean_pct_high", "std_pct_high", "dist_from_midpoint", "cronbach_alpha",
                ):
                    row[f"{axis}_{suffix}"] = ""
                continue
            row[f"{axis}_modal_letter"] = a.modal_letter
            row[f"{axis}_modal_freq"] = round(a.modal_freq, 4)
            row[f"{axis}_modal_freq_ci_lo"] = round(a.modal_freq_ci[0], 4)
            row[f"{axis}_modal_freq_ci_hi"] = round(a.modal_freq_ci[1], 4)
            row[f"{axis}_mean_pct_high"] = round(a.mean_pct_high, 2)
            row[f"{axis}_std_pct_high"] = round(a.std_pct_high, 2)
            row[f"{axis}_dist_from_midpoint"] = round(a.dist_from_midpoint, 2)
            row[f"{axis}_cronbach_alpha"] = (
                round(a.cronbach_alpha, 3) if a.cronbach_alpha is not None else ""
            )
        writer.writerow(row)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Synthesis paper skeleton
# --------------------------------------------------------------------------- #
def render_paper(
    stats: list[ModelStats],
    axes_order: list[str],
    baseline: dict | None = None,
) -> str:
    groups = _group_by_condition(stats)
    # Prefer a non-"default" (i.e. fixed-temperature, controlled) condition for
    # the headline cross-model comparison; providers' differing default
    # temperatures are a confound for comparing stability across models.
    primary_label, primary_group = groups[0]
    for label, group in groups:
        if group[0].temperature_condition != "default":
            primary_label, primary_group = label, group
            break
    comparable = [s for s in primary_group if s.n_valid > 0 and s.reliable]
    excluded = [s for s in primary_group if s.n_valid > 0 and not s.reliable]
    most_stable = max(comparable, key=lambda s: s.modal_type_freq, default=None)
    least_stable = min(comparable, key=lambda s: s.modal_type_freq, default=None)

    lines = [
        "# Do Large Language Models Have a Personality? An OEJTS Arena",
        "",
        f"_Draft generated {date.today().isoformat()}. Auto-filled from run data; "
        "edit the prose before publishing._",
        "",
        "## 1. Context & method",
        "",
        "We administered the Open Extended Jungian Type Scales (OEJTS 1.2), a "
        "free public-domain instrument covering the four MBTI dichotomies "
        "(E/I, S/N, T/F, J/P), to a portfolio of large language models. Each "
        "model answered the 32-item questionnaire in a single request, in "
        "English, with a system prompt instructing it to answer *as itself* "
        "rather than role-playing a character. Left/right anchor polarity was "
        "counterbalanced per run (half of each axis's items displayed with "
        "poles swapped, scored accordingly) to control for position/"
        "acquiescence bias. We repeated this independently N times per model "
        "under at least one fixed-temperature condition (to keep sampling "
        "entropy comparable across providers) to measure not just the type but "
        "its run-to-run **stability**, reported with 95% Wilson confidence "
        "intervals and against a Monte Carlo random-responder baseline.",
        "",
        f"Models tested: {', '.join(f'`{n}`' for n in sorted({s.model_name for s in stats}))}.",
        f"Primary cross-model comparison uses condition `{primary_label}`.",
        "",
    ]
    if baseline is not None:
        lines += [
            f"Random-responder baseline (uniform random answers, {baseline['n_trials']} "
            f"simulated models): expected modal-type stability "
            f"{baseline['modal_type_freq_mean']*100:.0f}% "
            f"(5th-95th pct {baseline['modal_type_freq_p05']*100:.0f}-"
            f"{baseline['modal_type_freq_p95']*100:.0f}%).",
            "",
        ]
    if excluded:
        lines += [
            f"_{len(excluded)} model(s) fell below the minimum valid-run threshold in "
            f"condition `{primary_label}` and are excluded from the superlatives below: "
            f"{', '.join(s.model_name for s in excluded)}._",
            "",
        ]

    lines += ["## 2. Results by model", ""]
    for s in primary_group:
        if s.n_valid == 0:
            lines.append(
                f"- **{s.model_name}** ({s.provider}): no valid runs "
                f"({s.n_invalid} invalid) — it declined or failed to answer in "
                "the required format, itself a notable behavior."
            )
            continue
        axis_items = [(ax, s.axes[ax]) for ax in axes_order if ax in s.axes]
        steadiest = max(axis_items, key=lambda kv: (kv[1].modal_freq, -kv[1].std_pct_high))[1]
        waviest = min(axis_items, key=lambda kv: (kv[1].modal_freq, -kv[1].std_pct_high))[1]
        reliability_flag = "" if s.reliable else " *(low N, low confidence)*"
        lines.append(
            f"- **{s.model_name}** ({s.provider}): modal type **{s.modal_type}** "
            f"in {s.modal_type_freq*100:.0f}% of runs (CI {_fmt_ci(s.modal_type_freq_ci)})"
            f"{reliability_flag}. Its firmest axis is "
            f"{steadiest.axis} (modal {steadiest.modal_letter}, "
            f"{steadiest.modal_freq*100:.0f}%); its most variable is "
            f"{waviest.axis} ({waviest.modal_freq*100:.0f}% modal, "
            f"σ={waviest.std_pct_high:.1f})."
        )
    lines += ["", "## 3. Cross-model comparison", ""]
    if most_stable and least_stable:
        lines += [
            f"- Most stable personality: **{most_stable.model_name}** "
            f"(type held in {most_stable.modal_type_freq*100:.0f}% of runs, "
            f"CI {_fmt_ci(most_stable.modal_type_freq_ci)}).",
            f"- Least stable personality: **{least_stable.model_name}** "
            f"(modal type only {least_stable.modal_type_freq*100:.0f}%, "
            f"CI {_fmt_ci(least_stable.modal_type_freq_ci)}).",
            "- Before treating these two as meaningfully different, check whether "
            "their confidence intervals actually overlap -- at N<=20 runs they "
            "often do.",
            "- TODO: discuss provider/family clustering and any size effects.",
            "",
        ]
    lines += [
        "## 4. Discussion & limits",
        "",
        "These numbers describe *how the models answer a self-report survey*, "
        "conditioned on prompt wording, language, and sampling temperature — not "
        "a validated personality trait. The MBTI/OEJTS framework is contested in "
        "psychology even for humans. The multi-run design, confidence intervals, "
        "random-responder baseline, and per-axis Cronbach's alpha are meant to "
        "foreground exactly this fragility: a type that changes from run to run, "
        "or an axis with low inter-item consistency, is a caution against "
        "over-interpreting any single result. See the README for the full list "
        "of caveats and prior work.",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Interactive HTML dashboard (self-contained, incl. vendored Chart.js)
# --------------------------------------------------------------------------- #
def render_dashboard(
    stats: list[ModelStats],
    axes_order: list[str],
    baseline: dict | None = None,
) -> str:
    groups = _group_by_condition(stats)
    payload = {
        "generated": date.today().isoformat(),
        "axes_order": axes_order,
        "palette": PALETTE,
        "baseline": baseline,
        "condition_order": [label for label, _ in groups],
        "conditions": {label: [s.to_dict() for s in group] for label, group in groups},
    }
    data_json = json.dumps(payload, ensure_ascii=False)
    chartjs_source = (_VENDOR_DIR / "chart.umd.min.js").read_text(encoding="utf-8")
    html = _DASHBOARD_TEMPLATE.replace("__DATA_JSON__", data_json)
    html = html.replace("__CHARTJS_SOURCE__", chartjs_source)
    return html


_DASHBOARD_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LLM MBTI Arena — Dashboard</title>
<script>
/* Vendored Chart.js (MIT License, Chart.js Contributors) -- inlined so this
   dashboard works fully offline as a single self-contained file. */
__CHARTJS_SOURCE__
</script>
<style>
  :root{
    --bg:#ffffff; --panel:#f6f7f9; --border:#e2e5ea; --text:#1a1d23;
    --muted:#5b616e; --accent:#0072B2;
  }
  @media (prefers-color-scheme: dark){
    :root{--bg:#0f1216;--panel:#181c22;--border:#2a2f38;--text:#e8eaed;--muted:#9aa1ad;}
  }
  *{box-sizing:border-box}
  body{margin:0;font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
    background:var(--bg);color:var(--text);line-height:1.5}
  header{padding:24px 20px 8px}
  h1{margin:0;font-size:1.5rem}
  .sub{color:var(--muted);font-size:.9rem;margin-top:4px}
  .tabs{display:flex;gap:8px;padding:12px 20px;flex-wrap:wrap;align-items:center}
  .tab{border:1px solid var(--border);background:var(--panel);color:var(--text);
    padding:6px 14px;border-radius:999px;cursor:pointer;font-size:.9rem}
  .tab.active{background:var(--accent);color:#fff;border-color:var(--accent)}
  main{padding:8px 20px 60px;max-width:1100px;margin:0 auto}
  .view{display:none}.view.active{display:block}
  table{border-collapse:collapse;width:100%;font-size:.86rem;overflow-x:auto;display:block}
  @media(min-width:720px){table{display:table}}
  th,td{border:1px solid var(--border);padding:7px 9px;text-align:left;white-space:nowrap}
  th{background:var(--panel)}
  td.type{font-weight:700;letter-spacing:1px}
  .bar{height:8px;border-radius:4px;background:var(--accent);display:inline-block;vertical-align:middle}
  .grid{display:grid;gap:20px;grid-template-columns:1fr}
  @media(min-width:680px){.grid{grid-template-columns:1fr 1fr}}
  .card{border:1px solid var(--border);border-radius:12px;padding:14px;background:var(--panel)}
  .card h3{margin:0 0 4px;font-size:1rem}
  .card .meta{color:var(--muted);font-size:.8rem;margin-bottom:10px}
  canvas{max-width:100%}
  select{background:var(--panel);color:var(--text);border:1px solid var(--border);
    border-radius:8px;padding:6px 10px;font-size:.9rem;margin:8px 0 16px}
  .chartbox{position:relative;height:320px}
  .legend{font-size:.8rem;color:var(--muted);margin-top:6px}
  .condbar{margin-left:auto;display:flex;align-items:center;gap:8px;font-size:.85rem;color:var(--muted)}
</style>
</head>
<body>
<header>
  <h1>LLM MBTI Arena</h1>
  <div class="sub" id="subtitle"></div>
</header>
<div class="tabs">
  <button class="tab active" data-view="compare">Compare all</button>
  <button class="tab" data-view="cards">Model cards</button>
  <button class="tab" data-view="focus">Single model</button>
  <span class="condbar" id="condBar"></span>
</div>
<main>
  <section id="compare" class="view active">
    <div id="table-wrap"></div>
    <div class="grid" style="margin-top:24px">
      <div class="card"><h3>Type stability by model</h3>
        <div class="chartbox"><canvas id="stabChart"></canvas></div>
        <div class="legend">Share of valid runs on each model's modal (most frequent) type, with 95% Wilson CI (hover a bar). Dashed reference line = random-responder baseline. Higher = more consistent "personality" -- but only meaningful above the baseline.</div>
      </div>
      <div class="card"><h3>Preference dispersion per axis</h3>
        <select id="axisPick"></select>
        <div class="chartbox"><canvas id="dispChart"></canvas></div>
        <div class="legend">Mean preference toward the second pole (I/N/T/P) per model. Hover a bar for σ across runs (not drawn as error bars).</div>
      </div>
    </div>
    <div class="grid" style="margin-top:20px">
      <div class="card" style="grid-column:1/-1"><h3>Stability vs. distance from midpoint (all models × axes)</h3>
        <div class="chartbox"><canvas id="scatterChart"></canvas></div>
        <div class="legend">Each point is one model's axis. X = |mean preference − 50|, i.e. how far from a coin flip the average answer is. Y = modal-letter frequency (stability). Points near the bottom-right corner (far from the midpoint but still "unstable") would be surprising; points near the top-left (near the midpoint but reported as "stable") are the dichotomization artifact: a hair's-breadth preference gets rounded to a confident-looking letter every run just because it rarely crosses 50%.</div>
      </div>
    </div>
  </section>

  <section id="cards" class="view">
    <div class="grid" id="cardGrid"></div>
  </section>

  <section id="focus" class="view">
    <select id="modelPick"></select>
    <div class="grid">
      <div class="card"><h3>Axis profile (radar)</h3>
        <div class="chartbox"><canvas id="focusRadar"></canvas></div>
      </div>
      <div class="card"><h3>Per-axis stability</h3>
        <div class="chartbox"><canvas id="focusStab"></canvas></div>
      </div>
    </div>
    <div id="focusInfo" class="card" style="margin-top:20px"></div>
  </section>
</main>

<script>
const DATA = __DATA_JSON__;
const AXES = DATA.axes_order;
const PAL = DATA.palette;
const BASELINE = DATA.baseline;
const CONDITIONS = DATA.condition_order;
let MODELS = DATA.conditions[CONDITIONS[0]];
const charts = {};

document.getElementById('subtitle').textContent =
  `${DATA.conditions[CONDITIONS[0]].length} models · generated ${DATA.generated}`;

/* ---- condition selector (only shown if >1 condition present) ---- */
if (CONDITIONS.length > 1) {
  const bar = document.getElementById('condBar');
  const label = document.createElement('span'); label.textContent = 'Condition:';
  const sel = document.createElement('select');
  CONDITIONS.forEach(c => { const o = document.createElement('option'); o.value = c; o.textContent = c; sel.appendChild(o); });
  sel.onchange = () => { MODELS = DATA.conditions[sel.value]; renderAll(); };
  bar.appendChild(label); bar.appendChild(sel);
}

function color(i){return PAL[i % PAL.length];}

/* ---- tabs ---- */
document.querySelectorAll('.tab').forEach(t=>t.onclick=()=>{
  document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
  document.querySelectorAll('.view').forEach(x=>x.classList.remove('active'));
  t.classList.add('active');
  document.getElementById(t.dataset.view).classList.add('active');
});

/* ---- comparison table ---- */
function buildTable(){
  let h = '<table><thead><tr><th>Model</th><th>Provider</th><th>Type</th><th>Stability (95% CI)</th><th>Reliable</th><th>Valid</th>';
  AXES.forEach(ax=>h+=`<th>${ax}</th>`);
  h+='</tr></thead><tbody>';
  MODELS.forEach((m,i)=>{
    const stab = Math.round(m.modal_type_freq*100);
    const ci = m.modal_type_freq_ci ? `[${Math.round(m.modal_type_freq_ci[0]*100)},${Math.round(m.modal_type_freq_ci[1]*100)}]` : '';
    h+=`<tr><td>${m.model_name}</td><td>${m.provider}</td>`+
       `<td class="type">${m.modal_type}</td>`+
       `<td><span class="bar" style="width:${stab}px;background:${color(i)}"></span> ${stab}% ${ci}</td>`+
       `<td>${m.reliable? '✓' : '⚠ low N'}</td>`+
       `<td>${m.n_valid}/${m.n_total}</td>`;
    AXES.forEach(ax=>{
      const a=m.axes[ax];
      if(!a){h+='<td>—</td>';return;}
      h+=`<td><b>${a.modal_letter}</b> ${Math.round(a.modal_freq*100)}%</td>`;
    });
    h+='</tr>';
  });
  h+='</tbody></table>';
  document.getElementById('table-wrap').innerHTML=h;
}

/* ---- stability bar (with baseline reference line + CI in tooltip) ---- */
function buildStab(){
  if(charts.stab) charts.stab.destroy();
  const datasets = [{label:'Type stability %',
    data:MODELS.map(m=>Math.round(m.modal_type_freq*100)),
    backgroundColor:MODELS.map((_,i)=>color(i))}];
  if (BASELINE) {
    datasets.push({
      type:'line', label:'Random baseline',
      data:MODELS.map(()=>Math.round(BASELINE.modal_type_freq_mean*100)),
      borderColor:'#999', borderDash:[6,4], pointRadius:0, borderWidth:2, fill:false,
    });
  }
  charts.stab = new Chart(document.getElementById('stabChart'),{
    type:'bar',
    data:{labels:MODELS.map(m=>m.model_name), datasets},
    options:{indexAxis:'y',
      plugins:{legend:{display: !!BASELINE},
        tooltip:{callbacks:{afterLabel:(c)=>{
          const m=MODELS[c.dataIndex]; if(!m || !m.modal_type_freq_ci) return '';
          return `95% CI [${Math.round(m.modal_type_freq_ci[0]*100)},${Math.round(m.modal_type_freq_ci[1]*100)}]`;
        }}}},
      scales:{x:{max:100,title:{display:true,text:'% of runs on modal type'}}}}
  });
}

/* ---- dispersion per axis ---- */
function buildDisp(){
  const sel=document.getElementById('axisPick');
  sel.innerHTML = '';
  AXES.forEach(ax=>{const o=document.createElement('option');o.value=ax;o.textContent=ax;sel.appendChild(o);});
  const draw=()=>{
    const ax=sel.value;
    if(charts.disp) charts.disp.destroy();
    charts.disp=new Chart(document.getElementById('dispChart'),{
      type:'bar',
      data:{labels:MODELS.map(m=>m.model_name),
        datasets:[{label:`mean pref toward ${ax[1]}`,
          data:MODELS.map(m=>m.axes[ax]?Math.round(m.axes[ax].mean_pct_high):null),
          backgroundColor:MODELS.map((_,i)=>color(i))}]},
      options:{plugins:{legend:{display:false},
        tooltip:{callbacks:{afterLabel:(c)=>{
          const m=MODELS[c.dataIndex];const a=m.axes[ax];
          return a?`σ=${a.std_pct_high}`:'';}}}},
        scales:{y:{min:0,max:100,title:{display:true,
          text:`0=first pole … 100=second pole`}}}}
    });
  };
  sel.onchange=draw; draw();
}

/* ---- scatter: stability vs distance from midpoint, per model x axis ---- */
function buildScatter(){
  if(charts.scatter) charts.scatter.destroy();
  const datasets = AXES.map((ax,ai)=>({
    label: ax,
    backgroundColor: color(ai),
    data: MODELS.filter(m=>m.axes[ax]).map(m=>({
      x: m.axes[ax].dist_from_midpoint,
      y: Math.round(m.axes[ax].modal_freq*100),
      _model: m.model_name,
    })),
  }));
  charts.scatter = new Chart(document.getElementById('scatterChart'),{
    type:'scatter',
    data:{datasets},
    options:{
      plugins:{tooltip:{callbacks:{label:(c)=>`${c.raw._model} (${c.dataset.label}): dist=${c.raw.x.toFixed(0)}, stability=${c.raw.y}%`}}},
      scales:{
        x:{min:0,max:50,title:{display:true,text:'distance from midpoint (0=coin flip, 50=unanimous)'}},
        y:{min:0,max:100,title:{display:true,text:'modal-letter stability %'}},
      }
    }
  });
}

/* ---- model cards ---- */
function buildCards(){
  const grid=document.getElementById('cardGrid');
  grid.innerHTML = '';
  MODELS.forEach((m,i)=>{
    const div=document.createElement('div');div.className='card';
    let axisTxt=AXES.map(ax=>{const a=m.axes[ax];return a?`${ax}: <b>${a.modal_letter}</b> ${Math.round(a.modal_freq*100)}%`:`${ax}: —`;}).join(' · ');
    div.innerHTML=`<h3>${m.model_name} <span style="color:${color(i)}">■</span></h3>`+
      `<div class="meta">${m.provider} · ${m.model_id}</div>`+
      `<div class="type" style="font-size:1.3rem;letter-spacing:2px">${m.modal_type}</div>`+
      `<div class="meta">stable ${Math.round(m.modal_type_freq*100)}% ${m.reliable?'':'⚠ low N'} · valid ${m.n_valid}/${m.n_total}</div>`+
      `<div style="font-size:.83rem;margin-top:8px">${axisTxt}</div>`;
    grid.appendChild(div);
  });
}

/* ---- focus view ---- */
function buildFocus(){
  const sel=document.getElementById('modelPick');
  sel.innerHTML = '';
  MODELS.forEach((m,i)=>{const o=document.createElement('option');o.value=i;o.textContent=m.model_name;sel.appendChild(o);});
  const draw=()=>{
    const m=MODELS[+sel.value];const i=+sel.value;
    if(charts.radar)charts.radar.destroy();
    if(charts.fstab)charts.fstab.destroy();
    charts.radar=new Chart(document.getElementById('focusRadar'),{
      type:'radar',
      data:{labels:AXES.map(ax=>`${ax} (→${ax[1]})`),
        datasets:[{label:m.model_name,
          data:AXES.map(ax=>m.axes[ax]?m.axes[ax].mean_pct_high:50),
          borderColor:color(i),backgroundColor:color(i)+'33',pointBackgroundColor:color(i)}]},
      options:{scales:{r:{min:0,max:100,ticks:{stepSize:25}}}}
    });
    charts.fstab=new Chart(document.getElementById('focusStab'),{
      type:'bar',
      data:{labels:AXES,datasets:[{label:'modal-letter frequency %',
        data:AXES.map(ax=>m.axes[ax]?Math.round(m.axes[ax].modal_freq*100):0),
        backgroundColor:color(i)}]},
      options:{plugins:{legend:{display:false}},scales:{y:{min:0,max:100}}}
    });
    let rows=AXES.map(ax=>{const a=m.axes[ax];if(!a)return `<li>${ax}: —</li>`;
      const alpha = (a.cronbach_alpha===null||a.cronbach_alpha===undefined)?'n/a':a.cronbach_alpha;
      return `<li><b>${ax}</b>: modal ${a.modal_letter} (${Math.round(a.modal_freq*100)}% of runs, `+
        `CI [${Math.round(a.modal_freq_ci[0]*100)},${Math.round(a.modal_freq_ci[1]*100)}]), `+
        `mean pref→${a.pole_high} ${Math.round(a.mean_pct_high)}% (σ=${a.std_pct_high}, dist=${a.dist_from_midpoint}pt), `+
        `α=${alpha}, letters ${Object.entries(a.letter_counts).map(([k,v])=>k+'×'+v).join(', ')}</li>`;}).join('');
    let inv = m.n_invalid? `<p class="meta">Invalid runs: ${m.n_invalid}. ${(m.invalid_reasons||[]).slice(0,2).join(' | ')}</p>`:'';
    document.getElementById('focusInfo').innerHTML=
      `<h3>${m.model_name} — ${m.modal_type} ${m.reliable?'':'⚠ low N'}</h3>`+
      `<div class="meta">${m.provider} · ${m.model_id} · answered w/o retry: ${Math.round(m.pct_first_attempt*100)}% · types seen: ${Object.entries(m.type_counts).map(([k,v])=>k+'×'+v).join(', ')||'—'}</div>`+
      `<ul style="font-size:.85rem">${rows}</ul>${inv}`;
  };
  sel.onchange=draw; draw();
}

function renderAll(){
  document.getElementById('subtitle').textContent = `${MODELS.length} models · generated ${DATA.generated}`;
  buildTable(); buildStab(); buildDisp(); buildScatter(); buildCards(); buildFocus();
}
renderAll();
</script>
</body>
</html>
"""


def write_reports(
    stats: list[ModelStats],
    axes_order: list[str],
    report_dir: str | Path,
    inst: Instrument | None = None,
    n_runs: int | None = None,
) -> dict[str, Path]:
    out = Path(report_dir)
    out.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()

    baseline = None
    if inst is not None and n_runs:
        baseline = random_baseline_stats(inst, n_runs)

    paths = {
        "markdown": out / f"comparatif_{today}.md",
        "dashboard": out / "dashboard.html",
        "csv": out / f"summary_{today}.csv",
        "paper": out / f"paper_{today}.md",
    }
    paths["markdown"].write_text(render_markdown(stats, axes_order, baseline=baseline), encoding="utf-8")
    paths["dashboard"].write_text(
        render_dashboard(stats, axes_order, baseline=baseline), encoding="utf-8"
    )
    paths["csv"].write_text(render_csv(stats, axes_order), encoding="utf-8")
    paths["paper"].write_text(render_paper(stats, axes_order, baseline=baseline), encoding="utf-8")
    return paths
