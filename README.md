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
   system prompt telling the model to answer **as itself**.
2. Repeats this **N=20 independent runs** at the model's **default temperature**.
3. Parses each strict-JSON reply, scores it into E/I · S/N · T/F · J/P letters
   with preference percentages, and stores the full raw run for auditability.
4. Aggregates per model: modal type, **type stability**, per-axis modal letter
   frequency, mean preference, and dispersion (σ).
5. Renders three artifacts: a comparative **markdown report**, an interactive
   **HTML dashboard**, and a **synthesis paper** skeleton auto-filled from data.

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

# 4. Run the experiment (calls the APIs), then build reports
python main.py run

# ...or run a subset
python main.py run --only claude-sonnet gpt

# 5. Re-build reports from existing runs without calling any API
python main.py report
```

Outputs land in:

- `data/runs/` — one JSON file per run (prompt, raw reply, parsed answers, score,
  token usage). Git-ignored; this is your raw, replayable data.
- `reports/dashboard.html` — the interactive dashboard (open in a browser).
- `reports/comparatif_<date>.md` — the comparative table + per-model detail.
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

---

## Architecture

```
config/
  models.yaml            # portfolio: provider, model_id, params (pinned)
  run_settings.yaml      # N runs, language, mode, seed, retries, concurrency
  instrument/oejts_32.yaml   # the 32 items, axis + direction (data, not code)
src/
  config.py              # load models + run settings
  instrument.py          # load + validate an instrument
  providers/
    base.py              # ProviderAdapter interface + GenerationResult
    openai_compatible.py # shared Chat-Completions adapter (configurable base_url)
    anthropic_provider.py, google_provider.py, …  # one adapter per provider
    registry.py          # provider name -> adapter class
  prompting/
    templates.py         # persona + one-shot prompt builder
    parser.py            # robust strict-JSON parsing with retries
  scoring/
    mbti_scorer.py       # deterministic, network-free, unit-tested
  runner/
    orchestrator.py      # model × N runs -> prompt -> parse -> score -> store
  report/
    aggregate.py         # per-model stats: modal type, stability, dispersion
    render.py            # markdown report + HTML dashboard + paper
tests/                   # scorer / parser / instrument / aggregate (no network)
main.py                  # CLI
```

**Design principles**

- **One adapter per provider behind a common interface** — adding a model is
  adding a config line (and an adapter file only for a genuinely new API shape).
- **The instrument is data.** Swap OEJTS for another questionnaire via YAML.
- **Every raw run is stored in full** — prompts, raw reply, parsed answers,
  score, usage — so scoring can be replayed and refusals stay auditable.
- **Scoring is deterministic and unit-tested**, fully separated from any network
  call.

### Methodology (fixed decisions)

- **No role-play.** The system prompt instructs each model to answer as *itself*,
  reported on every call.
- **Closed questions only.** Each item is a 1–5 semantic-differential rating
  between two anchors; the reply must be strict JSON — no free text. This kills
  role-play drift and makes parsing reliable.
- **One-shot.** All 32 items in a single request per run — cheaper, and avoids
  context drift over 32 turns.
- **N=20 runs at default temperature.** The headline statistic is run-to-run
  **stability**: a model that keeps the same type (low variance) vs. one that
  "changes personality" between runs (high variance) is itself a result.
- **Randomized item order** (deterministic per run seed) to blunt straight-lining;
  reverse-keying is handled by the scorer, not the prompt.
- **Fallback on refusal.** If a model refuses or breaks format, it is retried up
  to `max_retries` with an explicit reminder; if it still fails, the run is
  logged as **invalid** (a data point in itself) rather than fabricating scores.

### Scoring

Each item is tagged with an axis (`EI/SN/TF/JP`) and a direction (`keyed ±1`).
Per axis: orient every item toward its second-pole letter, sum, and compare to
the midpoint (`3 × n_items = 24`). Above midpoint → second letter (I/N/T/P),
otherwise first (E/S/F/J); ties break to the first letter. The **preference
percentage** (e.g. 62% E / 38% I) is kept alongside the letter. Over N runs we
report the modal letter per axis and its frequency — the stability measure.

---

## Methodological limits

Read these before drawing conclusions:

- The MBTI — and open derivatives like OEJTS — has **contested scientific
  validity** in psychology, even for humans. Applied to an LLM, the result
  measures a **prompt-conditioned text-output tendency shaped by training data**,
  not a "personality trait" in any real sense.
- Results are **sensitive to prompt wording, language, and temperature** — hence
  multiple runs and a stability metric rather than a single number.
- Prior academic work on MBTI-testing LLMs (e.g. GPT/InstructGPT/GPT-4) is
  documented at
  [github.com/Kali-Hac/ChatGPT-MBTI](https://github.com/Kali-Hac/ChatGPT-MBTI) —
  useful as a methodology reference, not code to reuse.

---

## Configuration reference

- `config/models.yaml` — set `enabled: false` to skip a model without deleting
  it; put per-model `params` (e.g. `max_tokens`) inline. Temperature is
  intentionally left unset (default temperature per model).
- `config/run_settings.yaml` — `n_runs`, `language`, `seed`, `max_retries`,
  `max_concurrency`, output/report dirs.
- `config/instrument/oejts_32.yaml` — the items and scoring metadata.

### Cost & re-runs

There is no hard request cap; token usage is captured per run in the raw JSON so
you can monitor spend (20 models × 20 runs = 400 base calls, more with
retries). Lower `n_runs` in `config/run_settings.yaml` for a cheap smoke test
before committing to a full run. Runs
are timestamped and versioned, so the same command can be re-executed later to
track how models drift over time — no code changes needed.

## Tests

```bash
python -m pytest -q
```

The scorer, parser, instrument loader, and report aggregation are covered by
network-free unit tests.

## License / attribution

OEJTS item text: Eric Jorgenson, *Open Extended Jungian Type Scales 1.2*, Open
Psychometrics (public domain). This repository is an independent research tool
and is not affiliated with or endorsed by The Myers-Briggs Company.
