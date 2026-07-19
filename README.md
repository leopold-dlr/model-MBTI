# LLM MBTI Arena

Administer the same personality questionnaire to a portfolio of large language
models, in a **reproducible** way, and compare the results — not just the
4-letter type each model lands on, but how **stable** that type is across
repeated runs. The goal is to surface each model's *own* default tendencies
(not a character it's asked to role-play) and to distil the findings into a
short synthesis paper.

> ⚠️ **This is an exploratory / playful experiment, not a psychometric
> measurement of AI.** See [Methodological limits](#methodological-limits).

---

## What it does

For every model in `config/models.yaml`, the pipeline:

1. Sends the **OEJTS** questionnaire (32 items, one-shot, English) with a
   system prompt telling the model to answer **as itself**. Left/right anchor
   polarity is **counterbalanced per run** (half of each axis's items are
   displayed with poles swapped, and scored accordingly) so a model with any
   systematic left/right or acquiescence bias doesn't land on the same letter
   regardless of its actual tendencies.
2. Repeats this **N=20 independent runs**, under one or more **temperature
   conditions** (provider default, and/or a fixed temperature shared by every
   model — see [Methodology](#methodology-fixed-decisions)) and optionally
   under multiple **prompt-variant** wordings.
3. Parses each strict-JSON reply, scores it into E/I · S/N · T/F · J/P letters
   with preference percentages, and stores the full raw run (both prompts,
   every attempt, not just the successful one) for auditability.
4. Aggregates per **(model, temperature condition, prompt variant)**: modal
   type, **type stability with a 95% Wilson confidence interval**, per-axis
   modal letter frequency, mean preference, dispersion (σ), and per-axis
   **Cronbach's alpha** (do that axis's items even covary for this model?).
   Stats are grouped by *experiment* (one per `run` invocation) so two
   separate launches are never silently pooled together.
5. Renders four artifacts: a comparative **markdown report**, an interactive
   **HTML dashboard** (fully self-contained, Chart.js vendored inline), a
   **CSV export**, and a **synthesis paper** skeleton auto-filled from data —
   all referencing a **Monte Carlo random-responder baseline** so "stable"
   numbers can be read against the floor a purely random answer pattern
   would produce.

### The instrument: OEJTS (not MBTI)

The official MBTI is a **trademarked, proprietary** instrument. Only the
underlying theory (Jung's dichotomies) is public. This project uses the
[**Open Extended Jungian Type Scales (OEJTS 1.2)**](https://openpsychometrics.org/tests/OEJTS/),
a free, public-domain, validated equivalent covering the same four dichotomies.

The instrument is **data, not code** (`config/instrument/oejts_32.yaml`), so you
can drop in another questionnaire following the same schema without touching the
runner or scorer.

---

## Quick start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Configure keys — copy the template and fill in the providers you want.
cp .env.example .env
$EDITOR .env

# 3. Preview the exact prompt (no API calls, no keys needed)
python main.py show-prompt
python main.py show-prompt --model gpt-5 --run-index 3   # exact per-run reproduction

# 4. Run the experiment (calls the APIs), then build reports
python main.py run

# ...or run a subset — exact name or family prefix (matches every gpt-5 tier)
python main.py run --only claude-sonnet gpt

# Interrupted mid-run? Resume instead of re-billing everything already valid.
python main.py run --resume 20260719T120000Z

# 5. Re-build reports from existing runs without calling any API
python main.py report                       # latest experiment only (default)
python main.py report --list-experiments    # see what's available
python main.py report --experiment 20260719T120000Z
```

Outputs land in:

- `data/runs/` — one JSON file per run (both prompts sent on every attempt,
  raw reply, parsed answers, score, token usage, item polarity, experiment
  id). Git-ignored; this is your raw, replayable data.
- `reports/dashboard.html` — the interactive dashboard (open in a browser;
  works fully offline, Chart.js is vendored inline).
- `reports/comparatif_<date>.md` — the comparative table + per-model detail.
- `reports/summary_<date>.csv` — the same stats, flat, for spreadsheets/stats tools.
- `reports/paper_<date>.md` — the synthesis paper draft.

You only ever need to supply API keys — no code changes required to run.

---

## The 20 models

The two biggest labs (Anthropic, OpenAI) get **3 size tiers** each
(flagship / mid / small); the next tier of major providers get **2** each; the
rest get 1. Beyond cross-provider comparison, the same-family tiers let you
check whether MBTI type and its **stability** shift with model size (e.g. does
`gpt-5-nano` flip type across runs more often than `gpt-5`?).

| # | Provider | Name in config | `model_id` | API key env |
|---|----------|-----------------|------------|--------------|
| 1 | Anthropic | `claude-opus` | `claude-opus-4-8` | `ANTHROPIC_API_KEY` |
| 2 | Anthropic | `claude-sonnet` | `claude-sonnet-5` | `ANTHROPIC_API_KEY` |
| 3 | Anthropic | `claude-haiku` | `claude-haiku-4-5-20251001` | `ANTHROPIC_API_KEY` |
| 4 | OpenAI | `gpt-5` | `gpt-5` | `OPENAI_API_KEY` |
| 5 | OpenAI | `gpt-5-mini` | `gpt-5-mini` | `OPENAI_API_KEY` |
| 6 | OpenAI | `gpt-5-nano` | `gpt-5-nano` | `OPENAI_API_KEY` |
| 7 | Google | `gemini-pro` | `gemini-3-pro` | `GOOGLE_API_KEY` |
| 8 | Google | `gemini-flash` | `gemini-3.5-flash` | `GOOGLE_API_KEY` |
| 9 | Mistral | `mistral-large` | `mistral-large-latest` | `MISTRAL_API_KEY` |
| 10 | Mistral | `mistral-small` | `mistral-small-latest` | `MISTRAL_API_KEY` |
| 11 | xAI | `grok-4` | `grok-4.3` | `XAI_API_KEY` |
| 12 | xAI | `grok-4-fast` | `grok-4.1-fast` | `XAI_API_KEY` |
| 13 | DeepSeek | `deepseek-flash` | `deepseek-v4-flash` (non-thinking) | `DEEPSEEK_API_KEY` |
| 14 | DeepSeek | `deepseek-pro` | `deepseek-v4-pro` (thinking) | `DEEPSEEK_API_KEY` |
| 15 | Alibaba | `qwen-max` | `qwen-max` | `DASHSCOPE_API_KEY` / `QWEN_API_KEY` |
| 16 | Alibaba | `qwen-plus` | `qwen-plus` | `DASHSCOPE_API_KEY` / `QWEN_API_KEY` |
| 17 | Meta (via Together) | `llama-maverick` | `meta-llama/Llama-4-Maverick-…` | `TOGETHER_API_KEY` |
| 18 | Meta (via Together) | `llama-scout` | `meta-llama/Llama-4-Scout-…` | `TOGETHER_API_KEY` |
| 19 | Moonshot | `kimi` | `kimi-k2.6` | `MOONSHOT_API_KEY` |
| 20 | Cohere | `command-a` | `command-a-03-2025` | `COHERE_API_KEY` |

> **Pin exact model ids, and re-verify them.** Providers rename and retire
> models on the order of weeks, not years — several ids above already replace
> ones that were retired mid-2026 (e.g. Kimi's `kimi-k2-0711-preview`,
> DeepSeek's `deepseek-chat`/`deepseek-reasoner`). `config/models.yaml` flags
> the entries worth double-checking against the provider's own model list
> right before a real run (`# verify:` comments). Each run record also stores
> the timestamp and the model id the provider echoed back, so results stay
> comparable over time even as the roster drifts.

A model whose API key is missing is **skipped with a warning** — you can run
just the subset you have keys for. No new env vars are needed versus a
10-model run: adding a same-provider tier only adds a `models.yaml` entry, not
a new key. Most providers are reached through their OpenAI-compatible
endpoints, so a single `openai` client covers Mistral, xAI, DeepSeek,
Moonshot, Qwen, Together and Cohere; Anthropic and Google use their native
SDKs.

> **Reasoning-capable models need a bigger token budget, and OpenAI's newer
> models need a different parameter name.** `gpt-5`/`gpt-5-mini`,
> `gemini-3-pro` and `grok-4.3` all spend part of their output budget on
> internal reasoning before any visible reply, so they're configured with
> `max_tokens: 4096` (vs. 2048 for the fast/small tiers) — a smaller budget
> risks a silently-truncated, unparseable reply that gets logged as invalid
> for no reason related to the model's "personality". Separately, gpt-5/
> o-series reject the legacy `max_tokens` field on `/chat/completions`
> entirely; `OpenAIAdapter` sends `max_completion_tokens` instead (see
> `src/providers/openai_provider.py`) while every other OpenAI-compatible
> provider still gets `max_tokens`.

---

## Architecture

```
config/
  models.yaml            # portfolio: provider, model_id, params (pinned)
  run_settings.yaml      # N runs, conditions, seed, retries, concurrency
  instrument/oejts_32.yaml   # the 32 items, axis + direction (data, not code)
  instrument/ipip50_bigfive_SCAFFOLD.yaml  # 2nd-instrument scaffold, NOT populated (see file header)
src/
  config.py              # load models + run settings
  instrument.py          # load + validate an instrument
  providers/
    base.py              # ProviderAdapter interface + GenerationResult
    openai_compatible.py # shared Chat-Completions adapter (configurable base_url)
    anthropic_provider.py, google_provider.py, …  # one adapter per provider
    registry.py          # provider name -> adapter class
  prompting/
    templates.py         # persona/prompt-variant + one-shot prompt builder + polarity flip_map
    parser.py            # robust strict-JSON parsing with retries
  scoring/
    mbti_scorer.py       # deterministic, network-free, unit-tested; keyed_overrides-aware
  runner/
    orchestrator.py      # model × condition × N runs -> prompt -> parse -> score -> store
  report/
    aggregate.py         # per-(model,condition) stats: modal type, CI, alpha, baseline
    render.py            # markdown report + HTML dashboard + CSV + paper
    vendor/chart.umd.min.js  # vendored Chart.js (MIT), so the dashboard works offline
tests/                   # scorer / parser / instrument / templates / orchestrator / aggregate (no network)
main.py                  # CLI
```

**Design principles**

- **One adapter per provider behind a common interface** — adding a model is
  adding a config line (and an adapter file only for a genuinely new API shape).
- **The instrument is data.** Swap OEJTS for another questionnaire via YAML —
  see the (intentionally unpopulated) IPIP scaffold for what that takes.
- **Every raw run is stored in full** — both prompts sent on *every* attempt
  (not just the successful one), raw reply, parsed answers, score, usage — so
  scoring can be replayed and refusals stay auditable.
- **Scoring is deterministic and unit-tested**, fully separated from any network
  call, and reproducible **across processes** (seeds are SHA-256-derived, not
  based on Python's per-process-salted `hash()`).

### Methodology (fixed decisions)

- **No role-play.** The system prompt instructs each model to answer as *itself*,
  reported on every call. An opt-in `prompt_variants` list
  (`run_settings.yaml`) lets you ablate the exact wording without touching code.
- **Closed questions only.** Each item is a 1–5 semantic-differential rating
  between two anchors; the reply must be strict JSON — no free text. This kills
  role-play drift and makes parsing reliable.
- **One-shot.** All 32 items in a single request per run — cheaper, and avoids
  context drift over 32 turns.
- **Polarity counterbalancing.** Every item in `oejts_32.yaml` orients its
  second (RIGHT) anchor toward the same set of letters (I/N/T/P). Left alone,
  a model with any systematic left/right or acquiescence bias would land on
  the same type regardless of its actual tendencies. Each run deterministically
  flips which side each pole is displayed on for half of each axis's items
  (`prompting.templates.flip_map`), inverting the effective `keyed` sign to
  match, and records exactly which items were flipped (`item_polarity`) so
  scoring is fully reproducible.
- **N=20 runs per (model, temperature condition, prompt variant).** The
  headline statistic is run-to-run **stability**: a model that keeps the same
  type (low variance) vs. one that "changes personality" between runs (high
  variance) is itself a result — reported with a **95% Wilson confidence
  interval**, since at N≤20 point estimates alone invite over-reading noise
  as signal.
- **Temperature is a controlled variable, not an afterthought.**
  `run_settings.yaml`'s `temperature_conditions` runs every model at the
  provider's own default (uncontrolled — different providers default to
  different sampling entropy) *and* at a fixed temperature shared by every
  model. **Only the fixed-temperature condition is valid for comparing
  stability across models/providers** — the default condition answers a
  different question ("how does this model behave as actually deployed").
- **Randomized item order** (deterministic per run seed) to blunt straight-lining;
  reverse-keying (canonical or polarity-flipped) is handled by the scorer, not
  the prompt.
- **Fallback on refusal, separated from infrastructure retries.** A malformed
  or refused reply is retried up to `max_retries` with an explicit reminder
  (and logged **invalid** if it still fails — a data point in itself, not
  fabricated scores). A transient API failure (network error, rate limit,
  5xx) gets its own `max_provider_retries` budget with backoff and no
  reminder, so a provider's rate limiting is never conflated with the model
  refusing to answer.
- **Reliability threshold.** A (model, condition) with fewer valid runs than
  `min_valid_runs` is flagged `reliable: false` in every report and excluded
  from the paper's "most/least stable" superlatives.

### Scoring

Each item is tagged with an axis (`EI/SN/TF/JP`) and a direction (`keyed ±1`),
which the per-run polarity flip can invert for that run only. Per axis:
orient every item toward its second-pole letter, sum, and compare to the
midpoint (`3 × n_items = 24`). Above midpoint → second letter (I/N/T/P),
otherwise first (E/S/F/J); ties break to the first letter. The **preference
percentage** (e.g. 62% E / 38% I) is kept alongside the letter. Over N runs we
report the modal letter per axis with its frequency and Wilson CI (the
stability measure), plus **Cronbach's alpha** — if an axis's 8 items don't
covary for a given model, its letter isn't measuring one coherent thing no
matter how "stable" it looks.

---

## Methodological limits

Read these before drawing conclusions:

- The MBTI — and open derivatives like OEJTS — has **contested scientific
  validity** in psychology, even for humans. Applied to an LLM, the result
  measures a **prompt-conditioned text-output tendency shaped by training data**,
  not a "personality trait" in any real sense.
- Results are **sensitive to prompt wording and language**. The wording
  ablation (`prompt_variants`) and a second language are supported but
  **opt-in** (disabled by default to keep the base run's cost fixed) —
  results from a single default wording in English should not be treated as
  wording-invariant unless you've actually run the ablation.
- **No independent (e.g. Big Five) instrument is included yet.** A second,
  differently-normed instrument would let you check convergent validity
  (do OEJTS "I" and an independent Extraversion score actually correlate?).
  `config/instrument/ipip50_bigfive_SCAFFOLD.yaml` documents the intended
  schema but ships with **zero items on purpose**: the canonical IPIP-50 item
  text could not be safely verified against a live source in the environment
  this was built in, and shipping guessed-from-memory psychometric item text
  under the IPIP name would be worse than shipping nothing. Populate it from
  [ipip.ori.org](https://ipip.ori.org/) before use.
- Even with Wilson CIs and a random-responder baseline, **N=20 runs per
  condition is still modest** — many pairwise model comparisons will have
  overlapping confidence intervals. Raise `n_runs` (cost scales linearly) if
  you need tighter comparisons.
- Prior academic work on MBTI-testing LLMs (e.g. GPT/InstructGPT/GPT-4) is
  documented at
  [github.com/Kali-Hac/ChatGPT-MBTI](https://github.com/Kali-Hac/ChatGPT-MBTI) —
  useful as a methodology reference, not code to reuse.

---

## Configuration reference

- `config/models.yaml` — set `enabled: false` to skip a model without deleting
  it; put per-model `params` (e.g. `max_tokens`) inline. Temperature is set
  per *condition* (see below), not per model.
- `config/run_settings.yaml` — `n_runs`, `language`, `seed`, `max_retries`,
  `max_provider_retries`, `max_concurrency`, `min_valid_runs`,
  `temperature_conditions` (list of `{label, value}`; `value: null` = provider
  default), `prompt_variants` (list of names from
  `src/prompting/templates.SYSTEM_PROMPTS`), output/report dirs.
- `config/instrument/oejts_32.yaml` — the items and scoring metadata.

### Cost & re-runs

There is no hard request cap; token usage is captured per run in the raw JSON so
you can monitor spend. With the shipped defaults (20 models × 20 runs × **2**
temperature conditions × 1 prompt variant) that's **800 base calls** before
retries — trim `temperature_conditions` to one entry to go back to 400, or
add prompt variants/a 2nd temperature to scale further. Lower `n_runs` for a
cheap smoke test before committing to a full run; **use `--resume
<experiment_id>` after an interrupted run instead of restarting from zero and
re-billing everything already valid.** Runs are timestamped, seeded, and
tagged with an `experiment_id`, so the same command can be re-executed later
to track how models drift over time — no code changes needed, and separate
launches are never silently pooled in reports (`report --list-experiments` /
`--experiment <id>` / `--all-experiments`).

## Tests

```bash
python -m pytest -q
```

The scorer (incl. polarity overrides), parser, instrument loader (incl. the
empty-scaffold rejection), prompt templates (incl. flip-map determinism and
prompt variants), orchestrator (incl. a cross-process seed-determinism
regression test and `--only` matching), and report aggregation (incl. Wilson
CIs, Cronbach's alpha, experiment/condition grouping, and the CSV/dashboard
renderers) are all covered by network-free unit tests.

## License / attribution

OEJTS item text: Eric Jorgenson, *Open Extended Jungian Type Scales 1.2*, Open
Psychometrics (public domain). This repository is an independent research tool
and is not affiliated with or endorsed by The Myers-Briggs Company.

The dashboard vendors [Chart.js](https://www.chartjs.org/) (MIT License,
Chart.js Contributors) inline so it runs fully offline — see
`src/report/vendor/CHARTJS_LICENSE.md`.
