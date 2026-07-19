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

# Categorical palette: 8 slots per mode, validated for colorblind-safe
# adjacent separation, lightness band, and normal-vision distinctness in BOTH
# light and dark modes (dark is the same hues re-stepped for the dark surface,
# not an automatic flip). The slot ORDER is the CVD-safety mechanism -- never
# reorder or extend it: models beyond slot 8 share a neutral gray in charts
# and are identified by hover labels / legend text / the table view instead
# (generating extra hues past 8 makes pairs indistinguishable under color
# blindness, which is worse than an honest gray).
PALETTE_LIGHT = [
    "#2a78d6", "#008300", "#e87ba4", "#eda100",
    "#1baf7a", "#eb6834", "#4a3aa7", "#e34948",
]
PALETTE_DARK = [
    "#3987e5", "#008300", "#d55181", "#c98500",
    "#199e70", "#d95926", "#9085e9", "#e66767",
]
PALETTE = PALETTE_LIGHT  # back-compat alias

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
        "palette_light": PALETTE_LIGHT,
        "palette_dark": PALETTE_DARK,
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


# Design notes (see the dataviz method this layout follows):
#  * Visible text is kept to one short line per card; the full methodology
#    explanations live in the (i) hover tooltips and the collapsed
#    "Methodology check" block, not in always-on paragraphs.
#  * The hero visual is the Trait map: each personality axis is a bipolar
#    slider between its two named poles, with one dot per model -- the
#    natural form for "where does each model sit between Extraverted and
#    Introverted", and the direct all-models-compared view.
#  * Consistency is a leaderboard of thin single-hue bars (magnitude job ->
#    one hue, not per-model colors) with the random-chance floor drawn as a
#    dashed line and the 95% CI as a whisker on each bar.
#  * Identity never rides on color alone: every dot/bar is named on hover,
#    a text legend lists the models, and the comparison table is the
#    always-available table view.
_DASHBOARD_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LLM Personality Arena</title>
<script>
/* Vendored Chart.js (MIT License, Chart.js Contributors) -- inlined so this
   dashboard works fully offline as a single self-contained file. */
__CHARTJS_SOURCE__
</script>
<style>
  :root{
    color-scheme: light;
    --page:#f9f9f7; --surface:#fcfcfb;
    --ink1:#0b0b0b; --ink2:#52514e; --muted:#898781;
    --grid:#e1e0d9; --baseline:#c3c2b7; --border:rgba(11,11,11,.10);
    --accent:#2a78d6;
  }
  @media (prefers-color-scheme: dark){
    :root{
      color-scheme: dark;
      --page:#0d0d0d; --surface:#1a1a19;
      --ink1:#ffffff; --ink2:#c3c2b7; --muted:#898781;
      --grid:#2c2c2a; --baseline:#383835; --border:rgba(255,255,255,.10);
      --accent:#3987e5;
    }
  }
  *{box-sizing:border-box}
  body{margin:0;font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
    background:var(--page);color:var(--ink1);line-height:1.45}
  header{padding:26px 20px 6px;max-width:1080px;margin:0 auto}
  h1{margin:0;font-size:1.45rem;letter-spacing:-.01em}
  .sub{color:var(--muted);font-size:.85rem;margin-top:3px}
  .tabs{display:flex;gap:8px;padding:14px 20px;flex-wrap:wrap;align-items:center;
    max-width:1080px;margin:0 auto}
  .tab{border:1px solid var(--border);background:var(--surface);color:var(--ink1);
    padding:6px 15px;border-radius:999px;cursor:pointer;font-size:.88rem}
  .tab.active{background:var(--accent);color:#fff;border-color:var(--accent)}
  .condbar{margin-left:auto;display:flex;align-items:center;gap:8px;
    font-size:.83rem;color:var(--muted)}
  main{padding:4px 20px 60px;max-width:1080px;margin:0 auto}
  .view{display:none}.view.active{display:block}
  .card{border:1px solid var(--border);border-radius:14px;padding:18px;
    background:var(--surface);margin-bottom:20px}
  .card h3{margin:0 0 2px;font-size:1.02rem;display:flex;align-items:center;gap:7px}
  .hint{color:var(--muted);font-size:.8rem;margin:0 0 14px}
  .info{display:inline-flex;align-items:center;justify-content:center;
    width:16px;height:16px;border-radius:50%;border:1px solid var(--baseline);
    color:var(--muted);font-size:.68rem;font-style:normal;cursor:help;flex:none}
  select{background:var(--surface);color:var(--ink1);border:1px solid var(--border);
    border-radius:8px;padding:6px 10px;font-size:.85rem}
  .grid2{display:grid;gap:20px;grid-template-columns:1fr}
  @media(min-width:680px){.grid2{grid-template-columns:1fr 1fr}}

  /* ---- trait map ---- */
  .axisrow{display:flex;align-items:center;gap:14px;margin:2px 0}
  .pole{width:190px;font-size:.82rem;color:var(--ink2)}
  .pole.l{text-align:right}
  .track{flex:1;position:relative;height:44px}
  .track::before{content:"";position:absolute;left:0;right:0;top:50%;height:2px;
    background:var(--grid);border-radius:1px}
  .track::after{content:"";position:absolute;left:50%;top:10px;bottom:10px;width:1px;
    background:var(--baseline)}
  .dot{position:absolute;width:13px;height:13px;border-radius:50%;
    border:2px solid var(--surface);transform:translate(-50%,-50%);cursor:default}
  .dot:hover{width:16px;height:16px;z-index:3}
  .chips{display:flex;flex-wrap:wrap;gap:6px 14px;margin-top:12px;
    font-size:.78rem;color:var(--ink2)}
  .chips i{display:inline-block;width:9px;height:9px;border-radius:50%;
    margin-right:5px;vertical-align:baseline}

  /* ---- consistency leaderboard ---- */
  .lb{position:relative}
  .lb-row{display:flex;align-items:center;gap:12px;margin:9px 0}
  .lb-name{width:190px;text-align:right;font-size:.85rem;
    white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .lb-flag{color:var(--muted);font-size:.72rem}
  .lb-track{flex:1;position:relative;height:16px}
  .lb-bar{position:absolute;top:3px;height:10px;background:var(--accent);
    border-radius:0 4px 4px 0}
  .lb-ci{position:absolute;top:7px;height:2px;background:var(--ink2);opacity:.65}
  .lb-ci::before,.lb-ci::after{content:"";position:absolute;top:-3px;width:2px;height:8px;
    background:inherit}
  .lb-ci::before{left:0}.lb-ci::after{right:0}
  .lb-base{position:absolute;top:-4px;bottom:-4px;width:0;
    border-left:2px dashed var(--baseline)}
  .lb-val{width:46px;font-size:.85rem;font-variant-numeric:tabular-nums}

  /* ---- table (scrolls inside its own wrapper, never the page) ---- */
  #table-wrap{overflow-x:auto}
  table{border-collapse:collapse;width:100%;font-size:.84rem}
  th,td{border:1px solid var(--grid);padding:7px 9px;text-align:left;white-space:nowrap}
  th{background:var(--page);color:var(--ink2);font-weight:600}
  td.type{font-weight:700;letter-spacing:1px}

  /* ---- model cards ---- */
  .card h3 .swatch{width:10px;height:10px;border-radius:50%;display:inline-block;flex:none}
  .headline{font-size:1.02rem;font-weight:700;margin:2px 0 8px}
  ul.traits{margin:0 0 8px;padding-left:1.1em;font-size:.87rem}
  ul.traits li{margin-bottom:2px}
  .example{font-size:.81rem;font-style:italic;color:var(--muted);margin:6px 0}
  .meta{color:var(--muted);font-size:.79rem}

  details{margin-bottom:20px}
  details summary{cursor:pointer;color:var(--muted);font-size:.85rem;padding:4px 2px}
  details .card{margin-top:10px}
  .chartbox{position:relative;height:320px}
  canvas{max-width:100%}
</style>
</head>
<body>
<header>
  <h1>LLM Personality Arena</h1>
  <div class="sub" id="subtitle"></div>
</header>
<div class="tabs">
  <button class="tab active" data-view="overview">Overview</button>
  <button class="tab" data-view="cards">Model cards</button>
  <button class="tab" data-view="focus">Model detail</button>
  <span class="condbar" id="condBar"></span>
</div>
<main>
  <section id="overview" class="view active">
    <div class="card">
      <h3>Trait map
        <span class="info" title="For every personality axis, each model is placed between the two opposite traits according to its average answers across all runs. The center line is a perfect 50/50 split. Hover a dot for the exact value. Where dots overlap they are nudged apart vertically.">i</span>
      </h3>
      <p class="hint">Each dot is a model — hover to identify. Center line = no lean either way.</p>
      <div id="traitMap"></div>
      <div class="chips" id="traitLegend"></div>
    </div>

    <div class="card">
      <h3>Consistency
        <span class="info" title="How often each model gave its own most frequent overall result across repeated, independent runs. The whisker on each bar is a 95% confidence interval: with few runs the true value could be anywhere in that range, so don't compare two models whose whiskers overlap. The dashed line is what a coin-flipping random answerer would score on average - a model is only showing a real, repeatable personality to the extent it beats that line.">i</span>
      </h3>
      <p class="hint">Same answer, run after run? Dashed line = random chance.</p>
      <div class="lb" id="leaderboard"></div>
    </div>

    <div class="card">
      <h3>Side by side</h3>
      <div id="table-wrap"></div>
    </div>

    <details>
      <summary>Methodology check: is "stable" real or a rounding artifact?</summary>
      <div class="card">
        <div class="chartbox"><canvas id="scatterChart"></canvas></div>
        <p class="hint" style="margin-top:10px">Each point is one model's axis. X = how far its average answer sits from a 50/50 split; Y = how often the same trait wins across runs. Points high on the left are the trap: a hair's-breadth preference that looks rock-solid only because it never crosses the midline.</p>
      </div>
    </details>
  </section>

  <section id="cards" class="view">
    <div class="grid2" id="cardGrid"></div>
  </section>

  <section id="focus" class="view">
    <select id="modelPick" style="margin-bottom:14px"></select>
    <div class="grid2">
      <div class="card"><h3>Trait profile</h3>
        <div class="chartbox"><canvas id="focusRadar"></canvas></div>
      </div>
      <div class="card"><h3>Per-trait consistency
        <span class="info" title="For each axis: the share of runs in which this model landed on the same side. 100% = it picked the same trait every single run.">i</span>
      </h3>
        <div class="chartbox"><canvas id="focusStab"></canvas></div>
      </div>
    </div>
    <div id="focusInfo" class="card"></div>
  </section>
</main>

<script>
const DATA = __DATA_JSON__;
const AXES = DATA.axes_order;
const BASELINE = DATA.baseline;
const CONDITIONS = DATA.condition_order;
const DARK = matchMedia('(prefers-color-scheme: dark)').matches;
const PAL = DARK ? DATA.palette_dark : DATA.palette_light;
let MODELS = DATA.conditions[CONDITIONS[0]];
const charts = {};

/* Models beyond the 8 validated palette slots share a neutral gray -- never
   generated hues; identity comes from hover labels and the legend text. */
function color(i){return i < PAL.length ? PAL[i] : '#898781';}
function mdBoldToHtml(s){return (s||'')
  .replace(/\*\*(.*?)\*\*/g, '<b>$1</b>')
  .replace(/`([^`]+)`/g, '$1');}
/* Long compound trait names ("Intellectual / imaginative") overflow chart
   label areas; charts use the first segment, tooltips keep the full name. */
function shortTrait(name){return (name||'').split(' / ')[0];}
function ink(name){return getComputedStyle(document.documentElement).getPropertyValue(name).trim();}
Chart.defaults.color = ink('--ink2');
Chart.defaults.borderColor = ink('--grid');
Chart.defaults.font.family = 'system-ui,-apple-system,"Segoe UI",Roboto,sans-serif';
Chart.defaults.animation = false;  /* a report artifact renders final values instantly */

document.querySelectorAll('.tab').forEach(t=>t.onclick=()=>{
  document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
  document.querySelectorAll('.view').forEach(x=>x.classList.remove('active'));
  t.classList.add('active');
  document.getElementById(t.dataset.view).classList.add('active');
});

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

/* ---- trait map: one bipolar slider per axis, one dot per model ---- */
function buildTraitMap(){
  const map = document.getElementById('traitMap');
  map.innerHTML = '';
  AXES.forEach(ax=>{
    const sample = MODELS.find(m=>m.axes[ax]);
    if(!sample) return;
    const a0 = sample.axes[ax];
    const low = a0.trait_low || a0.pole_low, high = a0.trait_high || a0.pole_high;
    const row = document.createElement('div'); row.className='axisrow';
    const track = document.createElement('div'); track.className='track';
    /* collision groups: dots within ~3% of each other get nudged apart vertically */
    const dots = MODELS.filter(m=>m.axes[ax]).map((m)=>({
      m, i: MODELS.indexOf(m), pct: m.axes[ax].mean_pct_high, sd: m.axes[ax].std_pct_high}));
    const groups = {};
    dots.forEach(d=>{const k=Math.round(d.pct/3); (groups[k]=groups[k]||[]).push(d);});
    Object.values(groups).forEach(g=>{
      g.forEach((d,j)=>{d.dy = (j-(g.length-1)/2)*11;});
    });
    dots.forEach(d=>{
      const s = document.createElement('span'); s.className='dot';
      s.style.left = d.pct+'%';
      s.style.top = 'calc(50% + '+(d.dy||0)+'px)';
      s.style.background = color(d.i);
      const toward = d.pct >= 50 ? high : low;
      const lean = Math.abs(d.pct-50) < 3 ? 'about even' :
        Math.round(d.pct >= 50 ? d.pct : 100-d.pct)+'% '+toward;
      s.title = d.m.model_name+' — '+lean+' (±'+d.sd.toFixed(0)+' across runs)';
      track.appendChild(s);
    });
    const pl = document.createElement('div'); pl.className='pole l'; pl.textContent = low;
    const pr = document.createElement('div'); pr.className='pole'; pr.textContent = high;
    row.appendChild(pl); row.appendChild(track); row.appendChild(pr);
    map.appendChild(row);
  });
  const legend = document.getElementById('traitLegend');
  legend.innerHTML = MODELS.map((m,i)=>
    `<span><i style="background:${color(i)}"></i>${m.model_name}</span>`).join('');
  if (MODELS.length > PAL.length) {
    legend.innerHTML += `<span style="color:var(--muted)">gray dots share a color — hover to identify</span>`;
  }
}

/* ---- consistency leaderboard: single-hue bars, CI whiskers, chance line ---- */
function buildLeaderboard(){
  const lb = document.getElementById('leaderboard');
  lb.innerHTML = '';
  const sorted = [...MODELS].sort((a,b)=>b.modal_type_freq-a.modal_type_freq);
  const basePct = BASELINE ? BASELINE.modal_type_freq_mean*100 : null;
  sorted.forEach(m=>{
    const row = document.createElement('div'); row.className='lb-row';
    const pct = Math.round(m.modal_type_freq*100);
    const ci = m.modal_type_freq_ci || [m.modal_type_freq, m.modal_type_freq];
    const flag = m.reliable ? '' : ' <span class="lb-flag" title="Too few runs for a firm conclusion">low N</span>';
    row.innerHTML =
      `<div class="lb-name" title="${m.model_name}">${m.model_name}${flag}</div>`+
      `<div class="lb-track">`+
        (basePct !== null ? `<div class="lb-base" style="left:${basePct}%" title="Random chance: ${Math.round(basePct)}%"></div>` : '')+
        `<div class="lb-bar" style="width:${pct}%"></div>`+
        `<div class="lb-ci" style="left:${(ci[0]*100).toFixed(1)}%;width:${((ci[1]-ci[0])*100).toFixed(1)}%" `+
          `title="95% CI: ${Math.round(ci[0]*100)}–${Math.round(ci[1]*100)}%"></div>`+
      `</div>`+
      `<div class="lb-val">${pct}%</div>`;
    lb.appendChild(row);
  });
}

/* ---- side-by-side table ---- */
function buildTable(){
  const anyTypeCode = MODELS.some(m=>m.has_type_code);
  let h = '<table><thead><tr><th>Model</th>'+(anyTypeCode?'<th>Type code</th>':'')+'<th>Consistency</th><th>Valid runs</th>';
  AXES.forEach(ax=>h+=`<th>${ax}</th>`);
  h+='</tr></thead><tbody>';
  MODELS.forEach((m)=>{
    const stab = Math.round(m.modal_type_freq*100);
    const ci = m.modal_type_freq_ci ? ` <span class="meta">[${Math.round(m.modal_type_freq_ci[0]*100)}–${Math.round(m.modal_type_freq_ci[1]*100)}]</span>` : '';
    h+=`<tr><td>${m.model_name}${m.reliable?'':' <span class="lb-flag">low N</span>'}</td>`+
       (anyTypeCode ? `<td class="type">${m.has_type_code?m.modal_type:'—'}</td>` : '')+
       `<td>${stab}%${ci}</td>`+
       `<td>${m.n_valid}/${m.n_total}</td>`;
    AXES.forEach(ax=>{
      const a=m.axes[ax];
      if(!a){h+='<td>—</td>';return;}
      h+=`<td><b>${a.modal_trait}</b> <span class="meta">${Math.round(a.modal_freq*100)}%</span></td>`;
    });
    h+='</tr>';
  });
  h+='</tbody></table>';
  document.getElementById('table-wrap').innerHTML=h;
}

/* ---- methodology scatter (collapsed by default) ---- */
function buildScatter(){
  if(charts.scatter) charts.scatter.destroy();
  const datasets = AXES.map((ax,ai)=>({
    label: ax,
    backgroundColor: color(ai),
    pointRadius: 5,
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
      plugins:{tooltip:{callbacks:{label:(c)=>`${c.raw._model} (${c.dataset.label}): ${c.raw.x.toFixed(0)}pt from even, ${c.raw.y}% consistent`}}},
      scales:{
        x:{min:0,max:50,title:{display:true,text:'distance from a 50/50 split'}},
        y:{min:0,max:100,title:{display:true,text:'% of runs on the same trait'}},
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
      div.innerHTML = `<h3><span class="swatch" style="background:${color(i)}"></span>${m.model_name}</h3>`+
        `<p class="meta">No valid runs to build a profile from.</p>`;
      grid.appendChild(div);
      return;
    }
    const bullets = AXES.filter(ax=>m.axes[ax]).map(ax=>`<li>${m.axes[ax].modal_trait}</li>`).join('');
    div.innerHTML=`<h3><span class="swatch" style="background:${color(i)}"></span>${m.model_name}</h3>`+
      `<p class="headline">${m.headline || ''}</p>`+
      `<ul class="traits">${bullets}</ul>`+
      (m.plain_example ? `<p class="example">${m.plain_example}</p>` : '')+
      `<p class="meta">${m.confidence_note || ''}</p>`;
    grid.appendChild(div);
  });
}

/* ---- model detail ---- */
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
      data:{labels:AXES.map(ax=>{const a=m.axes[ax]; const name=a?(a.trait_high||a.pole_high):ax; return `→ ${shortTrait(name)}`;}),
        datasets:[{label:m.model_name,
          data:AXES.map(ax=>m.axes[ax]?m.axes[ax].mean_pct_high:50),
          borderColor:color(i),backgroundColor:color(i)+'33',pointBackgroundColor:color(i)}]},
      options:{plugins:{legend:{display:false},
          tooltip:{callbacks:{label:(c)=>{const a=m.axes[AXES[c.dataIndex]];
            return `${Math.round(c.raw)}% toward ${a?(a.trait_high||a.pole_high):''}`;}}}},
        scales:{r:{min:0,max:100,ticks:{stepSize:25}}}}
    });
    charts.fstab=new Chart(document.getElementById('focusStab'),{
      type:'bar',
      data:{labels:AXES.map(ax=>m.axes[ax]?shortTrait(m.axes[ax].modal_trait):ax),
        datasets:[{label:'consistency %',
        data:AXES.map(ax=>m.axes[ax]?Math.round(m.axes[ax].modal_freq*100):0),
        backgroundColor:color(i),borderRadius:{topLeft:4,topRight:4},barThickness:26}]},
      options:{plugins:{legend:{display:false},
          tooltip:{callbacks:{label:(c)=>{const a=m.axes[AXES[c.dataIndex]];
            return `${a?a.modal_trait:''}: same side in ${c.raw}% of runs`;}}}},
        scales:{y:{min:0,max:100}}}
    });
    let rows=AXES.map(ax=>{const a=m.axes[ax];if(!a)return `<li>${ax}: —</li>`;
      const alpha = (a.cronbach_alpha===null||a.cronbach_alpha===undefined)?'n/a':a.cronbach_alpha;
      return `<li><b>${ax}: ${a.modal_trait}</b> (${Math.round(a.modal_freq*100)}% of runs, `+
        `CI [${Math.round(a.modal_freq_ci[0]*100)},${Math.round(a.modal_freq_ci[1]*100)}]), `+
        `mean pref→${a.trait_high||a.pole_high} ${Math.round(a.mean_pct_high)}% (σ=${a.std_pct_high}, dist=${a.dist_from_midpoint}pt), `+
        `α=${alpha}</li>`;}).join('');
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
      `<h3><span class="swatch" style="background:${color(i)}"></span>${m.model_name} ${m.reliable?'':'<span class="lb-flag">low N</span>'}</h3>`+
      `<p style="font-size:.95rem">${paragraph}</p>`+
      `<div class="meta">${m.provider} · ${m.model_id} · answered w/o retry: ${Math.round(m.pct_first_attempt*100)}%${typesSeen}</div>`+
      (examples ? `<p class="meta" style="margin-top:10px"><b>What it actually answered:</b></p><ul style="font-size:.84rem">${examples}</ul>` : '')+
      `<details><summary>Full per-axis numbers</summary><ul style="font-size:.84rem">${rows}</ul></details>${inv}`;
  };
  sel.onchange=draw; draw();
}

function renderAll(){
  document.getElementById('subtitle').textContent =
    `${MODELS.length} models · generated ${DATA.generated}`;
  buildTraitMap(); buildLeaderboard(); buildTable(); buildScatter(); buildCards(); buildFocus();
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
