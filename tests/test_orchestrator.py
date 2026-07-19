import os
import subprocess
import sys
from pathlib import Path

from src.runner.orchestrator import _matches_only, _run_seed

ROOT = Path(__file__).resolve().parent.parent


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
