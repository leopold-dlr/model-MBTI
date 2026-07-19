from types import SimpleNamespace

from main import _most_common_n_total


def test_most_common_n_total_uses_actual_data_not_live_config():
    """Regression test: the random-responder baseline must be sized from how
    many runs were actually loaded, not from run_settings.yaml's current
    n_runs -- those diverge whenever the config is edited (e.g. lowered for a
    smoke test, restored afterward) and `report` is re-run against older
    data without a fresh `run`."""
    stats = [
        SimpleNamespace(n_total=2),
        SimpleNamespace(n_total=2),
        SimpleNamespace(n_total=2),
    ]
    assert _most_common_n_total(stats) == 2


def test_most_common_n_total_empty_stats_returns_none():
    assert _most_common_n_total([]) is None
