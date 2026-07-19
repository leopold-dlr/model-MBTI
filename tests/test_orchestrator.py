import os
import subprocess
import sys
from pathlib import Path

from src.instrument import load_instrument
from src.runner.orchestrator import _matches_only, _run_seed, min_tokens_for_instrument

ROOT = Path(__file__).resolve().parent.parent
OEJTS = ROOT / "config" / "instrument" / "oejts_32.yaml"
IPIP50 = ROOT / "config" / "instrument" / "ipip50_bigfive.yaml"


def test_run_seed_is_deterministic_within_process():
    assert _run_seed(42, "claude-opus", 0) == _run_seed(42, "claude-opus", 0)


def test_run_seed_varies_by_model_and_run_index():
    assert _run_seed(42, "claude-opus", 0) != _run_seed(42, "gpt-5", 0)
    assert _run_seed(42, "claude-opus", 0) != _run_seed(42, "claude-opus", 1)


def test_run_seed_stable_across_processes_with_different_hash_seeds():
    """Regression test for the original bug: `hash((base_seed, model_name,
    run_index))` is salted per-process by PYTHONHASHSEED, so the same
    (base_seed, model, run_index) produced a DIFFERENT seed on every process
    restart -- silently breaking the "reproducible item order" guarantee.
    A same-process test can't catch this (the salt is fixed for the whole
    process), so this spawns two subprocesses with deliberately different
    PYTHONHASHSEED values and asserts they still agree.
    """
    code = (
        "import sys; sys.path.insert(0, %r); "
        "from src.runner.orchestrator import _run_seed; "
        "print(_run_seed(42, 'claude-opus', 0))" % str(ROOT)
    )
    results = []
    for hash_seed in ("0", "1"):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed)
        proc = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode == 0, proc.stderr
        results.append(proc.stdout.strip())
    assert results[0] == results[1], (
        "seed changed across processes with different PYTHONHASHSEED -- "
        "the SHA-256-based _run_seed must be process-independent"
    )


def test_matches_only_exact_and_prefix():
    assert _matches_only("gpt-5", ["gpt-5"])
    assert _matches_only("gpt-5-mini", ["gpt"])
    assert _matches_only("gpt-5-nano", ["gpt"])
    assert not _matches_only("claude-opus", ["gpt"])


def test_openrouter_adapter_registered_and_configured():
    from src.config import load_models
    from src.providers.registry import build_adapter

    adapter = build_adapter("openrouter", "anthropic/claude-sonnet-5", {"max_tokens": 2048})
    assert adapter.api_key_env == "OPENROUTER_API_KEY"
    assert adapter.base_url == "https://openrouter.ai/api/v1"
    # OpenRouter translates params per-provider itself; the adapter sends the
    # standard `max_tokens` field (unlike the direct-OpenAI adapter).
    assert adapter.max_tokens_param == "max_tokens"

    models = load_models(ROOT / "config" / "models_openrouter.yaml")
    assert len(models) == 20
    assert all(m.provider == "openrouter" for m in models)
    # Same portfolio names as the direct-API config, so --only and reports
    # stay comparable across the two configs.
    direct = load_models(ROOT / "config" / "models.yaml")
    assert {m.name for m in models} == {m.name for m in direct}


def test_min_tokens_scales_with_instrument_size():
    """Regression test: a live smoke test against IPIP-50 (50 items) hit a
    hard-coded 2048-token ceiling tuned for OEJTS (32 items) and came back
    with an empty/truncated response on 2 of 12 runs. The floor must scale
    with the instrument actually loaded, not assume OEJTS's item count."""
    oejts = load_instrument(OEJTS)
    ipip = load_instrument(IPIP50)
    assert min_tokens_for_instrument(ipip) > min_tokens_for_instrument(oejts)
    assert min_tokens_for_instrument(oejts) > 0
