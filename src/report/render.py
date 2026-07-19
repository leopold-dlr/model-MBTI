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


def _instrument_label(inst: Instrument | None) -> str:
    if inst is None:
        return "the instrument"
    return f"{inst.name} {inst.version}".strip()


def _instrument_validity_note(inst: Instrument | None) -> str:
    """Instrument-specific contested-validity caveat. Must not hardcode
    OEJTS/MBTI language: this report is also generated for IPIP-50 (Big
    Five), whose validity concerns and citation differ."""
    if inst is None or inst.name == "OEJTS":
        return (
            "the OEJTS is an open MBTI-style instrument whose validity is "
            "contested even for humans."
        )
    if inst.name == "IPIP-50-BigFive":
        return (
            "the Big Five / IPIP-50 is more broadly accepted in personality "
            "psychology than MBTI-style typologies, but its validity for "
            "humans still does not transfer automatically to an LLM."
        )
    return f"{_instrument_label(inst)} carries its own validity limits for humans, which do not automatically transfer to an LLM."


def _has_meaningful_type_code(s: ModelStats) -> bool:
    """False when every axis uses the generic "Lo"/"Hi" placeholder poles
    (e.g. IPIP-50 -- see that file's header: Big Five is dimensional, not
    typological, so its concatenated code has no real identity, unlike
    OEJTS's recognizable 4-letter MBTI-style codes). Showing a bare
    "HiHiHiHiHi" as if it meant something was a direct, valid complaint --
    this suppresses that display entirely rather than relying on wording to
    paper over a meaningless string."""
    if not s.axes:
        return False
    return not all(a.pole_low == "Lo" and a.pole_high == "Hi" for a in s.axes.values())


def _condition_display_label(temperature_condition: str, prompt_variant: str) -> str:
    """Plain-language label for a (temperature_condition, prompt_variant)
    group -- the raw internal label (e.g. "fixed_t1/default") was shown
    verbatim and unexplained, which is exactly the kind of jargon a
    non-technical reader has no way to decode."""
    label = (
        "Default settings (as normally deployed)"
        if temperature_condition == "default"
        else "Fixed temperature (fair comparison across models)"
    )
    if prompt_variant != "default":
        label += f" · prompt wording: {prompt_variant}"
    return label


# --------------------------------------------------------------------------- #
# Markdown comparative report
# --------------------------------------------------------------------------- #
def render_markdown(
    stats: list[ModelStats],
    axes_order: list[str],
    baseline: dict | None = None,
    inst: Instrument | None = None,
) -> str:
    groups = _group_by_condition(stats)
    lines = [
        "# LLM MBTI Arena — Comparative Report",
        "",
        f"_Generated {date.today().isoformat()} from {sum(s.n_total for s in stats)} "
        f"runs across {len({s.model_name for s in stats})} models, "
        f"{len(groups)} condition(s)._",
        "",
        "**How to read this report:** \"Personality profiles\" below describes, in "
        "plain language, what each model's answers suggest about its default "
        "characteristics -- with concrete examples of what it actually answered. "
        "\"Comparison table\" lines all models up side by side. \"Technical detail\" "
        "has the confidence intervals, dispersion, and inter-item consistency behind "
        "every claim in the profiles, for anyone who wants to check them.",
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

    lines += ["## Personality profiles", ""]
    for label, group in groups:
        cond_suffix = (
            f" — {_condition_display_label(group[0].temperature_condition, group[0].prompt_variant)}"
            if len(groups) > 1
            else ""
        )
        for s in group:
            lines += [f"### `{s.model_name}`{cond_suffix}", ""]
            if s.n_valid == 0:
                lines += [
                    f"No valid runs ({s.n_invalid} invalid) — it declined or failed to "
                    "answer in the required format on every attempt, itself a notable "
                    "behavior.",
                    "",
                ]
                continue
            lines += [_personality_paragraph(s, axes_order), ""]
            if s.example_items:
                lines.append("Concrete examples from its actual answers:")
                lines.append("")
                for ex in s.example_items:
                    lines.append(f"- {_fmt_example_item(ex)}")
                lines.append("")

    for label, group in groups:
        if len(groups) > 1:
            display_label = _condition_display_label(
                group[0].temperature_condition, group[0].prompt_variant
            )
            lines += [f"## Comparison table — {display_label}", ""]
            tc = group[0].temperature_condition
            if tc == "default":
                lines.append(
                    "_This is how each model behaves as normally deployed. Different "
                    "providers use different default settings, so don't use this table "
                    "alone to compare models to each other -- see the other test setting "
                    "below for that._"
                )
            else:
                lines.append(
                    "_Every model was run under the same settings here -- this is the fair, "
                    "apples-to-apples comparison across models._"
                )
            lines.append("")
        else:
            lines += ["## Comparison table", ""]

        any_type_code = any(_has_meaningful_type_code(s) for s in group)
        if not any_type_code:
            lines.append(
                "_This instrument is dimensional, not typological -- there's no meaningful "
                "4-letter code, only the trait columns below._"
            )
            lines.append("")
        header = "| Model | Provider |" + (" Type code |" if any_type_code else "") + " Stability (95% CI) | Reliable | Valid |"
        sep = "|---|---|" + ("---|" if any_type_code else "") + "---|---|---|"
        for axis in axes_order:
            header += f" {axis} |"
            sep += "---|"
        lines += [header, sep]

        for s in group:
            reliable_mark = "✓" if s.reliable else "⚠ low N"
            type_cell = f" **{s.modal_type}** |" if any_type_code else ""
            row = (
                f"| `{s.model_name}` | {s.provider} |{type_cell}"
                f" {s.modal_type_freq*100:.0f}% {_fmt_ci(s.modal_type_freq_ci)} "
                f"| {reliable_mark} | {s.n_valid}/{s.n_total} |"
            )
            for axis in axes_order:
                a = s.axes.get(axis)
                if a is None:
                    row += " — |"
                    continue
                row += f" {a.modal_trait} ({a.modal_freq*100:.0f}%) |"
            lines.append(row)
        lines.append("")

    lines += [
        "## Technical detail (for the curious)",
        "",
        "The numbers behind each profile above: confidence intervals, dispersion, "
        "and per-axis inter-item consistency. Skip this section if the profiles "
        "and comparison table already answered your question.",
        "",
    ]
    for label, group in groups:
        cond_suffix = (
            f" — {_condition_display_label(group[0].temperature_condition, group[0].prompt_variant)}"
            if len(groups) > 1
            else ""
        )
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
                "",
            ]
            axis_rows = [
                "| Axis | Trait (stability, 95% CI) | Mean preference | Dispersion (σ) | Distance from midpoint | Cronbach α |",
                "|---|---|---|---|---|---|",
            ]
            for axis in axes_order:
                a = s.axes.get(axis)
                if a is None:
                    continue
                alpha_str = f"{a.cronbach_alpha:.2f}" if a.cronbach_alpha is not None else "n/a"
                pref_name = a.trait_high or a.pole_high
                axis_rows.append(
                    f"| {axis} | **{a.modal_trait}** ({a.modal_freq*100:.0f}%, "
                    f"{_fmt_ci(a.modal_freq_ci)}) | {pref_name}={a.mean_pct_high:.0f}% "
                    f"| {a.std_pct_high:.1f} | {a.dist_from_midpoint:.0f}pt | {alpha_str} |"
                )
            lines += axis_rows
            lines.append("")
            if s.n_invalid:
                lines.append(_invalid_summary(s))
            lines.append("")

    instrument_note = _instrument_validity_note(inst)
    lines += [
        "## Methodological caveats",
        "",
        f"See the project README. In short: {instrument_note} Applied to an "
        "LLM it measures a *prompt-conditioned text-output tendency*, not a "
        "personality trait. Treat this as exploratory, not psychometric. A low "
        "or undefined Cronbach's alpha on an axis means that axis's items don't "
        "covary for that model -- its letter is not measuring one coherent "
        "thing, whatever its apparent run-to-run stability.",
        "",
    ]
    return "\n".join(lines)


def _join_names(names: list[str]) -> str:
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + f", and {names[-1]}"


def _personality_paragraph(s: ModelStats, axes_order: list[str]) -> str:
    """A readable personality-profile paragraph -- the actual point of the
    whole exercise ("what characteristics does this model show"), written so
    it's readable without knowing what a Wilson CI or Cronbach's alpha is.
    The Technical detail section further down has the numbers behind every
    claim made here, for anyone who wants to check them."""
    axis_items = [(ax, s.axes[ax]) for ax in axes_order if ax in s.axes]
    traits_str = _join_names([a.modal_trait for _, a in axis_items])

    if not s.reliable:
        confidence = (
            f"This is based on only {s.n_valid} run(s) so far -- far too few to call "
            "any of it a stable trait yet; read it as a first impression, not a result."
        )
    elif s.modal_type_freq == 1.0:
        confidence = f"It gave this same profile in every one of its {s.n_valid} runs in this test."
    else:
        confidence = (
            f"It gave this profile in {s.modal_type_freq*100:.0f}% of its {s.n_valid} runs "
            f"(95% CI {_fmt_ci(s.modal_type_freq_ci)}) -- check whether that range overlaps "
            "with another model's before calling a difference between them meaningful."
        )

    clearest_sent = weakest_sent = ""
    if axis_items:
        clearest_axis, clearest = max(axis_items, key=lambda kv: kv[1].dist_from_midpoint)
        weakest_axis, weakest = min(axis_items, key=lambda kv: kv[1].dist_from_midpoint)
        clearest_pref = (
            clearest.mean_pct_high if clearest.modal_letter == clearest.pole_high
            else 100 - clearest.mean_pct_high
        )
        clearest_sent = (
            f" Its strongest, clearest lean is toward {clearest.modal_trait} "
            f"({clearest_pref:.0f}% preference)."
        )
        if weakest_axis != clearest_axis:
            wlow = weakest.trait_low or weakest.pole_low
            whigh = weakest.trait_high or weakest.pole_high
            weakest_sent = (
                f" It's much closer to a coin flip between {wlow} and {whigh} "
                f"({weakest.dist_from_midpoint:.0f} points from dead even) -- don't read a "
                "firm lean into that particular trait."
            )

    code_note = (
        f" (Sometimes shortened elsewhere to the code **{s.modal_type}**.)"
        if _has_meaningful_type_code(s)
        else ""
    )
    return (
        f"**`{s.model_name}`** comes across as {traits_str}.{clearest_sent}{weakest_sent} "
        f"{confidence}{code_note}"
    )


def _headline_traits(s: ModelStats, axes_order: list[str], n: int = 2) -> str:
    """A short 2-3 word "vibe" from the model's clearest (most decisive)
    traits -- the one thing someone should remember and be able to repeat,
    not a data point. Picked by distance from the midpoint, same ranking
    the full paragraph uses for its "clearest trait" sentence."""
    axis_items = [(ax, s.axes[ax]) for ax in axes_order if ax in s.axes]
    if not axis_items:
        return ""
    ranked = sorted(axis_items, key=lambda kv: kv[1].dist_from_midpoint, reverse=True)
    return _join_names([a.modal_trait for _, a in ranked[:n]])


def _plain_example(s: ModelStats) -> str:
    """The single clearest example answer, stripped of item ids/axis codes/
    percentages -- just what it said, for a card-sized glance rather than
    the fuller per-item breakdown in the profile paragraph's example list."""
    if not s.example_items:
        return ""
    phrase = s.example_items[0]["phrase"].replace("**", "")
    return f"For example, it {phrase}."


def _confidence_note(s: ModelStats) -> str:
    if not s.reliable:
        return f"Based on a small test ({s.n_valid} runs) -- a first impression, not a firm conclusion."
    if s.modal_type_freq == 1.0:
        return f"Consistent across all {s.n_valid} runs in this test."
    return f"Showed up in {s.modal_type_freq*100:.0f}% of this test's runs."


def _fmt_example_item(ex: dict) -> str:
    """Render one concrete item-level example: what the model actually
    answered, in its own words (the instrument's item text), not just an
    aggregate statistic -- grounds the profile paragraph in something real."""
    return (
        f'Item {ex["item_id"]} (axis {ex["axis"]}): it {ex["phrase"]} '
        f'({ex["pct"]:.0f}% preference, averaged over {ex["n"]} run(s)).'
    )


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
    inst: Instrument | None = None,
) -> str:
    groups = _group_by_condition(stats)
    instrument_label = _instrument_label(inst)
    n_items = len(inst.items) if inst is not None else None
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
        f"# Do Large Language Models Have a Personality? — {instrument_label} Arena",
        "",
        f"_Draft generated {date.today().isoformat()}. Auto-filled from run data; "
        "edit the prose before publishing._",
        "",
        "## 1. Context & method",
        "",
        f"We administered the {instrument_label} instrument"
        + (f" ({n_items} items, covering the axes {', '.join(axes_order)})" if n_items else "")
        + ", a free public-domain questionnaire, to a portfolio of large "
        "language models. Each model answered the full questionnaire in a "
        "single request, in English, with a system prompt instructing it to "
        "answer *as itself* rather than role-playing a character. Left/right "
        "anchor polarity was counterbalanced per run (half of each axis's "
        "items displayed with poles swapped, scored accordingly) to control "
        "for position/acquiescence bias. We repeated this independently N "
        "times per model under at least one fixed-temperature condition (to "
        "keep sampling entropy comparable across providers) to measure not "
        "just the type but its run-to-run **stability**, reported with 95% "
        "Wilson confidence intervals and against a Monte Carlo "
        "random-responder baseline.",
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
            f"{reliability_flag}. Its firmest trait is "
            f"{steadiest.modal_trait} ({steadiest.axis}, "
            f"{steadiest.modal_freq*100:.0f}%); its most variable is "
            f"{waviest.axis} ({waviest.pole_low}/{waviest.trait_low or waviest.pole_low} vs "
            f"{waviest.pole_high}/{waviest.trait_high or waviest.pole_high}, "
            f"{waviest.modal_freq*100:.0f}% modal, σ={waviest.std_pct_high:.1f})."
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
    else:
        lines += [
            f"_No comparison shown: every model in condition `{primary_label}` is "
            f"below `min_valid_runs` ({len(excluded)} model(s): "
            f"{', '.join(s.model_name for s in excluded) or 'none with valid runs at all'}). "
            "This is expected for a small smoke test (e.g. n_runs=2) -- raise `n_runs` "
            "and re-run before drawing any most/least-stable conclusion._",
            "",
        ]
    lines += [
        "## 4. Discussion & limits",
        "",
        "These numbers describe *how the models answer a self-report survey*, "
        "conditioned on prompt wording, language, and sampling temperature — not "
        f"a validated personality trait. {_instrument_validity_note(inst)} "
        "The multi-run design, confidence intervals, "
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

    def _model_payload(s: ModelStats) -> dict:
        d = s.to_dict()
        has_valid = s.n_valid > 0
        d["personality_paragraph"] = _personality_paragraph(s, axes_order) if has_valid else None
        d["headline"] = _headline_traits(s, axes_order) if has_valid else None
        d["plain_example"] = _plain_example(s) if has_valid else None
        d["confidence_note"] = _confidence_note(s) if has_valid else None
        d["has_type_code"] = _has_meaningful_type_code(s) if has_valid else False
        return d

    payload = {
        "generated": date.today().isoformat(),
        "axes_order": axes_order,
        "palette": PALETTE,
        "baseline": baseline,
        "condition_order": [label for label, _ in groups],
        "condition_labels": {
            label: _condition_display_label(group[0].temperature_condition, group[0].prompt_variant)
            for label, group in groups
        },
        "conditions": {label: [_model_payload(s) for s in group] for label, group in groups},
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
  .card .headline{font-size:1.05rem;font-weight:700;margin:2px 0 8px}
  .card ul.traits{margin:0 0 8px;padding-left:1.1em;font-size:.88rem}
  .card ul.traits li{margin-bottom:2px}
  .card .example{font-size:.82rem;font-style:italic;color:var(--muted);margin:6px 0}
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
    <div class="grid" style="margin-top:20px">
      <div class="card" style="grid-column:1/-1"><h3>All models × all traits, side by side</h3>
        <div class="chartbox" style="height:400px"><canvas id="allTraitsChart"></canvas></div>
        <div class="legend">Mean preference (0-100%) toward each axis's second pole, every model and every trait on one chart -- this is the direct "compare them to each other" view. Hover a bar for the exact trait name and value.</div>
      </div>
    </div>
    <div class="grid" style="margin-top:20px">
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
  const label = document.createElement('span'); label.textContent = 'Test setting:';
  const sel = document.createElement('select');
  CONDITIONS.forEach(c => {
    const o = document.createElement('option');
    o.value = c; o.textContent = DATA.condition_labels[c] || c;
    sel.appendChild(o);
  });
  sel.onchange = () => { MODELS = DATA.conditions[sel.value]; renderAll(); };
  bar.appendChild(label); bar.appendChild(sel);
}

function color(i){return PAL[i % PAL.length];}
function mdBoldToHtml(s){return (s||'').replace(/\*\*(.*?)\*\*/g, '<b>$1</b>');}

/* ---- tabs ---- */
document.querySelectorAll('.tab').forEach(t=>t.onclick=()=>{
  document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
  document.querySelectorAll('.view').forEach(x=>x.classList.remove('active'));
  t.classList.add('active');
  document.getElementById(t.dataset.view).classList.add('active');
});

/* ---- comparison table ---- */
function buildTable(){
  const anyTypeCode = MODELS.some(m=>m.has_type_code);
  let h = '<table><thead><tr><th>Model</th><th>Provider</th>'+(anyTypeCode?'<th>Type code</th>':'')+'<th>Stability (95% CI)</th><th>Reliable</th><th>Valid</th>';
  AXES.forEach(ax=>h+=`<th>${ax}</th>`);
  h+='</tr></thead><tbody>';
  MODELS.forEach((m,i)=>{
    const stab = Math.round(m.modal_type_freq*100);
    const ci = m.modal_type_freq_ci ? `[${Math.round(m.modal_type_freq_ci[0]*100)},${Math.round(m.modal_type_freq_ci[1]*100)}]` : '';
    h+=`<tr><td>${m.model_name}</td><td>${m.provider}</td>`+
       (anyTypeCode ? `<td class="type">${m.has_type_code?m.modal_type:'—'}</td>` : '')+
       `<td><span class="bar" style="width:${stab}px;background:${color(i)}"></span> ${stab}% ${ci}</td>`+
       `<td>${m.reliable? '✓' : '⚠ low N'}</td>`+
       `<td>${m.n_valid}/${m.n_total}</td>`;
    AXES.forEach(ax=>{
      const a=m.axes[ax];
      if(!a){h+='<td>—</td>';return;}
      h+=`<td><b>${a.modal_trait}</b> ${Math.round(a.modal_freq*100)}%</td>`;
    });
    h+='</tr>';
  });
  h+='</tbody></table>';
  document.getElementById('table-wrap').innerHTML=h;
}

/* ---- all models x all traits, one chart -- the direct cross-model comparison ---- */
function buildAllTraits(){
  if(charts.allTraits) charts.allTraits.destroy();
  const datasets = AXES.map((ax,ai)=>{
    const sample = MODELS.find(m=>m.axes[ax]);
    const traitName = sample ? (sample.axes[ax].trait_high || sample.axes[ax].pole_high) : ax;
    return {
      label: `${ax}: ${traitName}`,
      data: MODELS.map(m=>m.axes[ax]?Math.round(m.axes[ax].mean_pct_high):null),
      backgroundColor: color(ai),
    };
  });
  charts.allTraits = new Chart(document.getElementById('allTraitsChart'),{
    type:'bar',
    data:{labels:MODELS.map(m=>m.model_name), datasets},
    options:{
      plugins:{tooltip:{callbacks:{label:(c)=>`${c.dataset.label}: ${c.raw}%`}}},
      scales:{y:{min:0,max:100,title:{display:true,text:'mean preference toward the labeled trait (%)'}}}
    }
  });
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
    const sample = MODELS.find(m=>m.axes[ax]);
    const lowName = sample ? (sample.axes[ax].trait_low || sample.axes[ax].pole_low) : ax;
    const highName = sample ? (sample.axes[ax].trait_high || sample.axes[ax].pole_high) : ax;
    if(charts.disp) charts.disp.destroy();
    charts.disp=new Chart(document.getElementById('dispChart'),{
      type:'bar',
      data:{labels:MODELS.map(m=>m.model_name),
        datasets:[{label:`mean preference toward ${highName}`,
          data:MODELS.map(m=>m.axes[ax]?Math.round(m.axes[ax].mean_pct_high):null),
          backgroundColor:MODELS.map((_,i)=>color(i))}]},
      options:{plugins:{legend:{display:false},
        tooltip:{callbacks:{afterLabel:(c)=>{
          const m=MODELS[c.dataIndex];const a=m.axes[ax];
          return a?`σ=${a.std_pct_high}`:'';}}}},
        scales:{y:{min:0,max:100,title:{display:true,
          text:`0=${lowName} … 100=${highName}`}}}}
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
    if (m.n_valid === 0) {
      div.innerHTML = `<h3>${m.model_name} <span style="color:${color(i)}">■</span></h3>`+
        `<p class="meta">No valid runs to build a profile from.</p>`;
      grid.appendChild(div);
      return;
    }
    const bullets = AXES.filter(ax=>m.axes[ax]).map(ax=>`<li>${m.axes[ax].modal_trait}</li>`).join('');
    div.innerHTML=`<h3>${m.model_name} <span style="color:${color(i)}">■</span></h3>`+
      `<p class="headline">${m.headline || ''}</p>`+
      `<ul class="traits">${bullets}</ul>`+
      (m.plain_example ? `<p class="example">${m.plain_example}</p>` : '')+
      `<p class="meta">${m.confidence_note || ''}</p>`;
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
      data:{labels:AXES.map(ax=>{const a=m.axes[ax]; const name=a?(a.trait_high||a.pole_high):ax; return `${ax}: → ${name}`;}),
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
      return `<li><b>${ax}: ${a.modal_trait}</b> (${Math.round(a.modal_freq*100)}% of runs, `+
        `CI [${Math.round(a.modal_freq_ci[0]*100)},${Math.round(a.modal_freq_ci[1]*100)}]), `+
        `mean pref→${a.trait_high||a.pole_high} ${Math.round(a.mean_pct_high)}% (σ=${a.std_pct_high}, dist=${a.dist_from_midpoint}pt), `+
        `α=${alpha}, letters ${Object.entries(a.letter_counts).map(([k,v])=>k+'×'+v).join(', ')}</li>`;}).join('');
    let inv = m.n_invalid? `<p class="meta">Invalid runs: ${m.n_invalid}. ${(m.invalid_reasons||[]).slice(0,2).join(' | ')}</p>`:'';
    const paragraph = mdBoldToHtml(m.personality_paragraph || 'No valid runs to build a profile from.');
    const examples = (m.example_items||[]).map(ex=>
      `<li>Item ${ex.item_id} (${ex.axis}): it ${mdBoldToHtml(ex.phrase)} `+
      `(${ex.pct.toFixed(0)}% preference, avg. over ${ex.n} run(s)).</li>`
    ).join('');
    const typesSeen = m.has_type_code
      ? ` · codes seen: ${Object.entries(m.type_counts).map(([k,v])=>k+'×'+v).join(', ')||'—'}`
      : '';
    document.getElementById('focusInfo').innerHTML=
      `<h3>${m.model_name} ${m.reliable?'':'⚠ low N'}</h3>`+
      `<p style="font-size:1rem">${paragraph}</p>`+
      `<div class="meta">${m.provider} · ${m.model_id} · answered w/o retry: ${Math.round(m.pct_first_attempt*100)}%${typesSeen}</div>`+
      (examples ? `<p style="font-size:.85rem;margin-top:10px"><b>Concrete examples of what it answered:</b></p><ul style="font-size:.85rem">${examples}</ul>` : '')+
      `<p style="font-size:.85rem;margin-top:10px"><b>Full per-axis detail:</b></p>`+
      `<ul style="font-size:.85rem">${rows}</ul>${inv}`;
  };
  sel.onchange=draw; draw();
}

function renderAll(){
  document.getElementById('subtitle').textContent = `${MODELS.length} models · generated ${DATA.generated}`;
  buildTable(); buildAllTraits(); buildStab(); buildDisp(); buildScatter(); buildCards(); buildFocus();
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
    paths["markdown"].write_text(
        render_markdown(stats, axes_order, baseline=baseline, inst=inst), encoding="utf-8"
    )
    paths["dashboard"].write_text(
        render_dashboard(stats, axes_order, baseline=baseline), encoding="utf-8"
    )
    paths["csv"].write_text(render_csv(stats, axes_order), encoding="utf-8")
    paths["paper"].write_text(
        render_paper(stats, axes_order, baseline=baseline, inst=inst), encoding="utf-8"
    )
    return paths
