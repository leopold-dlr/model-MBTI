"""Render aggregated stats into (1) a comparative markdown report, (2) an
interactive self-contained HTML dashboard, and (3) a synthesis paper skeleton.

All three are generated from the same aggregated data, so re-running the
pipeline refreshes them without any manual editing.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from .aggregate import ModelStats

# Colorblind-safe categorical palette (Okabe-Ito), used for per-model series.
PALETTE = [
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7",
    "#56B4E9", "#F0E442", "#999999", "#332288", "#AA4499",
]


# --------------------------------------------------------------------------- #
# Markdown comparative report
# --------------------------------------------------------------------------- #
def render_markdown(stats: list[ModelStats], axes_order: list[str]) -> str:
    lines = [
        "# LLM MBTI Arena — Comparative Report",
        "",
        f"_Generated {date.today().isoformat()} from {sum(s.n_total for s in stats)} "
        f"runs across {len(stats)} models._",
        "",
        "## Summary table",
        "",
        "Stability = share of valid runs landing on the model's modal type. "
        "Per-axis cells show the modal letter and its run frequency; the "
        "percentage is the mean preference toward that pole.",
        "",
    ]
    header = "| Model | Provider | Modal type | Stability | Valid |"
    sep = "|---|---|---|---|---|"
    for axis in axes_order:
        header += f" {axis} |"
        sep += "---|"
    lines += [header, sep]

    for s in stats:
        row = (
            f"| `{s.model_name}` | {s.provider} | **{s.modal_type}** "
            f"| {s.modal_type_freq*100:.0f}% | {s.n_valid}/{s.n_total} |"
        )
        for axis in axes_order:
            a = s.axes.get(axis)
            if a is None:
                row += " — |"
                continue
            freq = a.modal_freq * 100
            pct = a.mean_pct_high if a.modal_letter == a.pole_high else (100 - a.mean_pct_high)
            row += f" {a.modal_letter} ({freq:.0f}%, {pct:.0f}%) |"
        lines.append(row)

    lines += ["", "## Per-model detail", ""]
    for s in stats:
        lines += [f"### `{s.model_name}` — {s.provider} / `{s.model_id}`", ""]
        if s.n_valid == 0:
            lines += [
                f"- No valid runs ({s.n_invalid} invalid).",
                _invalid_summary(s),
                "",
            ]
            continue
        lines += [
            f"- Modal type: **{s.modal_type}** (stable in {s.modal_type_freq*100:.0f}% of runs)",
            f"- Valid runs: {s.n_valid}/{s.n_total}"
            + (f" ({s.n_invalid} invalid)" if s.n_invalid else ""),
            f"- Types observed: {_fmt_counts(s.type_counts)}",
        ]
        for axis in axes_order:
            a = s.axes.get(axis)
            if a is None:
                continue
            lines.append(
                f"- {axis}: modal **{a.modal_letter}** "
                f"({a.modal_freq*100:.0f}% of runs); "
                f"mean pref toward {a.pole_high}={a.mean_pct_high:.0f}% "
                f"(σ={a.std_pct_high:.1f}); letters {_fmt_counts(a.letter_counts)}"
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
        "personality trait. Treat this as exploratory, not psychometric.",
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
    from collections import Counter

    c = Counter(r[:80] for r in s.invalid_reasons)
    reasons = "; ".join(f"{k} (×{v})" for k, v in c.most_common(3))
    return f"- Invalid runs: {reasons}"


# --------------------------------------------------------------------------- #
# Synthesis paper skeleton
# --------------------------------------------------------------------------- #
def render_paper(stats: list[ModelStats], axes_order: list[str]) -> str:
    valid_stats = [s for s in stats if s.n_valid > 0]
    most_stable = max(valid_stats, key=lambda s: s.modal_type_freq, default=None)
    least_stable = min(valid_stats, key=lambda s: s.modal_type_freq, default=None)

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
        "English, at its own default temperature, with a system prompt "
        "instructing it to answer *as itself* rather than role-playing a "
        "character. We repeated this independently N times per model to measure "
        "not just the type but its run-to-run **stability**.",
        "",
        f"Models tested: {', '.join(f'`{s.model_name}`' for s in stats)}.",
        "",
        "## 2. Results by model",
        "",
    ]
    for s in stats:
        if s.n_valid == 0:
            lines.append(
                f"- **{s.model_name}** ({s.provider}): no valid runs "
                f"({s.n_invalid} invalid) — it declined or failed to answer in "
                "the required format, itself a notable behavior."
            )
            continue
        # Identify the model's most and least stable axes.
        axis_items = [(ax, s.axes[ax]) for ax in axes_order if ax in s.axes]
        # Break modal-frequency ties by preference dispersion so the two picks
        # differ even when every axis is equally stable.
        steadiest = max(axis_items, key=lambda kv: (kv[1].modal_freq, -kv[1].std_pct_high))[1]
        waviest = min(axis_items, key=lambda kv: (kv[1].modal_freq, -kv[1].std_pct_high))[1]
        lines.append(
            f"- **{s.model_name}** ({s.provider}): modal type **{s.modal_type}** "
            f"in {s.modal_type_freq*100:.0f}% of runs. Its firmest axis is "
            f"{steadiest.axis} (modal {steadiest.modal_letter}, "
            f"{steadiest.modal_freq*100:.0f}%); its most variable is "
            f"{waviest.axis} ({waviest.modal_freq*100:.0f}% modal, "
            f"σ={waviest.std_pct_high:.1f})."
        )
    lines += ["", "## 3. Cross-model comparison", ""]
    if most_stable and least_stable:
        lines += [
            f"- Most stable personality: **{most_stable.model_name}** "
            f"(type held in {most_stable.modal_type_freq*100:.0f}% of runs).",
            f"- Least stable personality: **{least_stable.model_name}** "
            f"(modal type only {least_stable.modal_type_freq*100:.0f}%).",
            "- TODO: discuss provider/family clustering and any size effects.",
            "",
        ]
    lines += [
        "## 4. Discussion & limits",
        "",
        "These numbers describe *how the models answer a self-report survey*, "
        "conditioned on prompt wording, language, and sampling temperature — not "
        "a validated personality trait. The MBTI/OEJTS framework is contested in "
        "psychology even for humans. The multi-run design and stability metric "
        "are meant to foreground exactly this fragility: a type that changes "
        "from run to run is a caution against over-interpreting any single "
        "result. See the README for the full list of caveats and prior work.",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Interactive HTML dashboard (self-contained)
# --------------------------------------------------------------------------- #
def render_dashboard(stats: list[ModelStats], axes_order: list[str]) -> str:
    payload = {
        "generated": date.today().isoformat(),
        "axes_order": axes_order,
        "palette": PALETTE,
        "models": [s.to_dict() for s in stats],
    }
    data_json = json.dumps(payload, ensure_ascii=False)
    # Chart.js is loaded from a CDN; the data itself is embedded so the file is
    # a single shareable artifact.
    return _DASHBOARD_TEMPLATE.replace("__DATA_JSON__", data_json)


_DASHBOARD_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LLM MBTI Arena — Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
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
  .tabs{display:flex;gap:8px;padding:12px 20px;flex-wrap:wrap}
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
</div>
<main>
  <section id="compare" class="view active">
    <div id="table-wrap"></div>
    <div class="grid" style="margin-top:24px">
      <div class="card"><h3>Type stability by model</h3>
        <div class="chartbox"><canvas id="stabChart"></canvas></div>
        <div class="legend">Share of valid runs on each model's modal (most frequent) type. Higher = more consistent "personality".</div>
      </div>
      <div class="card"><h3>Preference dispersion per axis</h3>
        <select id="axisPick"></select>
        <div class="chartbox"><canvas id="dispChart"></canvas></div>
        <div class="legend">Mean preference toward the second pole (I/N/T/P) with ±1σ error bars across runs.</div>
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
const MODELS = DATA.models;
const PAL = DATA.palette;
const charts = {};

document.getElementById('subtitle').textContent =
  `${MODELS.length} models · generated ${DATA.generated}`;

function color(i){return PAL[i % PAL.length];}
function pctToward(a){ // preference toward pole_high (second letter), 0-100
  return a ? a.mean_pct_high : 50;
}

/* ---- tabs ---- */
document.querySelectorAll('.tab').forEach(t=>t.onclick=()=>{
  document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
  document.querySelectorAll('.view').forEach(x=>x.classList.remove('active'));
  t.classList.add('active');
  document.getElementById(t.dataset.view).classList.add('active');
});

/* ---- comparison table ---- */
function buildTable(){
  let h = '<table><thead><tr><th>Model</th><th>Provider</th><th>Type</th><th>Stability</th><th>Valid</th>';
  AXES.forEach(ax=>h+=`<th>${ax}</th>`);
  h+='</tr></thead><tbody>';
  MODELS.forEach((m,i)=>{
    const stab = Math.round(m.modal_type_freq*100);
    h+=`<tr><td>${m.model_name}</td><td>${m.provider}</td>`+
       `<td class="type">${m.modal_type}</td>`+
       `<td><span class="bar" style="width:${stab}px;background:${color(i)}"></span> ${stab}%</td>`+
       `<td>${m.n_valid}/${m.n_total}</td>`;
    AXES.forEach(ax=>{
      const a=m.axes[ax];
      if(!a){h+='<td>—</td>';return;}
      const pct = a.modal_letter===a.pole_high? a.mean_pct_high : (100-a.mean_pct_high);
      h+=`<td><b>${a.modal_letter}</b> ${Math.round(a.modal_freq*100)}% · ${Math.round(pct)}%</td>`;
    });
    h+='</tr>';
  });
  h+='</tbody></table>';
  document.getElementById('table-wrap').innerHTML=h;
}

/* ---- stability bar ---- */
function buildStab(){
  new Chart(document.getElementById('stabChart'),{
    type:'bar',
    data:{labels:MODELS.map(m=>m.model_name),
      datasets:[{label:'Type stability %',
        data:MODELS.map(m=>Math.round(m.modal_type_freq*100)),
        backgroundColor:MODELS.map((_,i)=>color(i))}]},
    options:{indexAxis:'y',plugins:{legend:{display:false}},
      scales:{x:{max:100,title:{display:true,text:'% of runs on modal type'}}}}
  });
}

/* ---- dispersion per axis ---- */
function buildDisp(){
  const sel=document.getElementById('axisPick');
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
          text:`0=${AXES.length?'first pole':''} … 100=second pole`}}}}
    });
  };
  sel.onchange=draw; draw();
}

/* ---- model cards ---- */
function buildCards(){
  const grid=document.getElementById('cardGrid');
  MODELS.forEach((m,i)=>{
    const div=document.createElement('div');div.className='card';
    let axisTxt=AXES.map(ax=>{const a=m.axes[ax];return a?`${ax}: <b>${a.modal_letter}</b> ${Math.round(a.modal_freq*100)}%`:`${ax}: —`;}).join(' · ');
    div.innerHTML=`<h3>${m.model_name} <span style="color:${color(i)}">■</span></h3>`+
      `<div class="meta">${m.provider} · ${m.model_id}</div>`+
      `<div class="type" style="font-size:1.3rem;letter-spacing:2px">${m.modal_type}</div>`+
      `<div class="meta">stable ${Math.round(m.modal_type_freq*100)}% · valid ${m.n_valid}/${m.n_total}</div>`+
      `<div style="font-size:.83rem;margin-top:8px">${axisTxt}</div>`;
    grid.appendChild(div);
  });
}

/* ---- focus view ---- */
function buildFocus(){
  const sel=document.getElementById('modelPick');
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
      return `<li><b>${ax}</b>: modal ${a.modal_letter} (${Math.round(a.modal_freq*100)}% of runs), `+
        `mean pref→${a.pole_high} ${Math.round(a.mean_pct_high)}% (σ=${a.std_pct_high}), `+
        `letters ${Object.entries(a.letter_counts).map(([k,v])=>k+'×'+v).join(', ')}</li>`;}).join('');
    let inv = m.n_invalid? `<p class="meta">Invalid runs: ${m.n_invalid}. ${(m.invalid_reasons||[]).slice(0,2).join(' | ')}</p>`:'';
    document.getElementById('focusInfo').innerHTML=
      `<h3>${m.model_name} — ${m.modal_type}</h3>`+
      `<div class="meta">${m.provider} · ${m.model_id} · types seen: ${Object.entries(m.type_counts).map(([k,v])=>k+'×'+v).join(', ')||'—'}</div>`+
      `<ul style="font-size:.85rem">${rows}</ul>${inv}`;
  };
  sel.onchange=draw; draw();
}

buildTable(); buildStab(); buildDisp(); buildCards(); buildFocus();
</script>
</body>
</html>
"""


def write_reports(
    stats: list[ModelStats], axes_order: list[str], report_dir: str | Path
) -> dict[str, Path]:
    out = Path(report_dir)
    out.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    paths = {
        "markdown": out / f"comparatif_{today}.md",
        "dashboard": out / "dashboard.html",
        "paper": out / f"paper_{today}.md",
    }
    paths["markdown"].write_text(render_markdown(stats, axes_order), encoding="utf-8")
    paths["dashboard"].write_text(render_dashboard(stats, axes_order), encoding="utf-8")
    paths["paper"].write_text(render_paper(stats, axes_order), encoding="utf-8")
    return paths
