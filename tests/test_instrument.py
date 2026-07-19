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
    expected = {
        "EI": ("E", "I"),
        "SN": ("S", "N"),
        "TF": ("F", "T"),
        "JP": ("J", "P"),
    }
    for axis, (low, high) in expected.items():
        assert inst.axes[axis]["pole_low"] == low
        assert inst.axes[axis]["pole_high"] == high


def test_axis_trait_names_present():
    inst = load_instrument(OEJTS)
    # trait_low/trait_high are the human-readable names used in reports, on
    # top of the bare pole_low/pole_high letters used for the type string.
    expected = {
        "EI": ("Extraversion", "Introversion"),
        "SN": ("Sensing", "Intuition"),
        "TF": ("Feeling", "Thinking"),
        "JP": ("Judging", "Perceiving"),
    }
    for axis, (low, high) in expected.items():
        assert inst.axes[axis]["trait_low"] == low
        assert inst.axes[axis]["trait_high"] == high


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


def test_ipip50_trait_names_present_and_readable():
    # pole_low/pole_high stay generic ("Lo"/"Hi") on purpose (Big Five is
    # dimensional, not typological -- see file header), but trait_low/high
    # must carry real, non-empty trait names for reports to be legible.
    inst = load_instrument(IPIP50)
    for axis in inst.type_order:
        poles = inst.axes[axis]
        assert poles["pole_low"] == "Lo"
        assert poles["pole_high"] == "Hi"
        assert poles["trait_low"] and poles["trait_low"] != "Lo"
        assert poles["trait_high"] and poles["trait_high"] != "Hi"


def test_ipip50_keyed_counts_match_source():
    # Documented on the source page: 5+/5- Extraversion, 6+/4- Agreeableness;
    # the rest derived the same way from the same page's per-item (+/-) key.
    inst = load_instrument(IPIP50)
    expected = {"EXTR": (5, 5), "AGRE": (6, 4), "CONS": (6, 4), "STAB": (2, 8), "INTL": (7, 3)}
    for axis, (plus, minus) in expected.items():
        items = inst.items_for_axis(axis)
        assert sum(1 for it in items if it.keyed == 1) == plus, axis
        assert sum(1 for it in items if it.keyed == -1) == minus, axis
