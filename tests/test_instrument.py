from pathlib import Path

from src.instrument import load_instrument

INSTRUMENT = Path(__file__).resolve().parent.parent / "config" / "instrument" / "oejts_32.yaml"


def test_loads_32_items():
    inst = load_instrument(INSTRUMENT)
    assert len(inst.items) == 32
    assert set(inst.item_ids()) == set(range(1, 33))


def test_axes_balanced():
    inst = load_instrument(INSTRUMENT)
    for axis in inst.type_order:
        assert len(inst.items_for_axis(axis)) == 8, axis


def test_axis_poles():
    inst = load_instrument(INSTRUMENT)
    assert inst.axes["EI"] == {"pole_low": "E", "pole_high": "I"}
    assert inst.axes["SN"] == {"pole_low": "S", "pole_high": "N"}
    assert inst.axes["TF"] == {"pole_low": "F", "pole_high": "T"}
    assert inst.axes["JP"] == {"pole_low": "J", "pole_high": "P"}


def test_scale_bounds():
    inst = load_instrument(INSTRUMENT)
    assert inst.scale_min == 1
    assert inst.scale_max == 5
    assert inst.scale_midpoint == 3
