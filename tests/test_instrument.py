from pathlib import Path

import pytest

from src.instrument import load_instrument

ROOT = Path(__file__).resolve().parent.parent
OEJTS = ROOT / "config" / "instrument" / "oejts_32.yaml"
IPIP50 = ROOT / "config" / "instrument" / "ipip50_bigfive.yaml"


def test_loads_32_items():
    inst = load_instrument(OEJTS)
    assert len(inst.items) == 32
    assert set(inst.item_ids()) == set(range(1, 33))


def test_axes_balanced():
    inst = load_instrument(OEJTS)
    for axis in inst.type_order:
        assert len(inst.items_for_axis(axis)) == 8, axis


def test_axis_poles():
    inst = load_instrument(OEJTS)
    assert inst.axes["EI"] == {"pole_low": "E", "pole_high": "I"}
    assert inst.axes["SN"] == {"pole_low": "S", "pole_high": "N"}
    assert inst.axes["TF"] == {"pole_low": "F", "pole_high": "T"}
    assert inst.axes["JP"] == {"pole_low": "J", "pole_high": "P"}


def test_scale_bounds():
    inst = load_instrument(OEJTS)
    assert inst.scale_min == 1
    assert inst.scale_max == 5
    assert inst.scale_midpoint == 3


def test_empty_instrument_rejected(tmp_path):
    bad = tmp_path / "empty.yaml"
    bad.write_text(
        "name: empty\nversion: '1'\nsource: ''\n"
        "scale: {min: 1, max: 5}\n"
        "axes: {AX: {pole_low: Lo, pole_high: Hi}}\n"
        "type_order: [AX]\nitems: []\n"
    )
    with pytest.raises(ValueError, match="no items"):
        load_instrument(bad)


def test_ipip50_loads_50_items_5_axes_of_10():
    inst = load_instrument(IPIP50)
    assert len(inst.items) == 50
    assert set(inst.item_ids()) == set(range(1, 51))
    assert inst.type_order == ["EXTR", "AGRE", "CONS", "STAB", "INTL"]
    for axis in inst.type_order:
        assert len(inst.items_for_axis(axis)) == 10, axis


def test_ipip50_keyed_counts_match_source():
    # Documented on the source page: 5+/5- Extraversion, 6+/4- Agreeableness;
    # the rest derived the same way from the same page's per-item (+/-) key.
    inst = load_instrument(IPIP50)
    expected = {"EXTR": (5, 5), "AGRE": (6, 4), "CONS": (6, 4), "STAB": (2, 8), "INTL": (7, 3)}
    for axis, (plus, minus) in expected.items():
        items = inst.items_for_axis(axis)
        assert sum(1 for it in items if it.keyed == 1) == plus, axis
        assert sum(1 for it in items if it.keyed == -1) == minus, axis
