from pathlib import Path

import pytest

from src.instrument import Instrument, Item, load_instrument
from src.scoring.mbti_scorer import score_answers, score_axis

INSTRUMENT = Path(__file__).resolve().parent.parent / "config" / "instrument" / "oejts_32.yaml"


def _all(inst, value):
    return {it.id: value for it in inst.items}


def test_all_max_gives_second_poles():
    inst = load_instrument(INSTRUMENT)
    res = score_answers(inst, _all(inst, 5))
    assert res.type == "INTP"
    for a in res.axes.values():
        assert a.pct_high == 100.0
        assert a.letter_pct == 100.0


def test_all_min_gives_first_poles():
    inst = load_instrument(INSTRUMENT)
    res = score_answers(inst, _all(inst, 1))
    assert res.type == "ESFJ"
    for a in res.axes.values():
        assert a.pct_high == 0.0


def test_neutral_ties_break_to_first_pole():
    inst = load_instrument(INSTRUMENT)
    res = score_answers(inst, _all(inst, 3))
    # sum == midpoint => pole_low wins on every axis
    assert res.type == "ESFJ"
    for a in res.axes.values():
        assert a.pct_high == 50.0


def test_missing_answer_raises():
    inst = load_instrument(INSTRUMENT)
    answers = _all(inst, 3)
    answers.pop(1)
    with pytest.raises(KeyError):
        score_answers(inst, answers)


def test_reverse_keyed_item():
    # Two-item single-axis instrument: item 1 normal, item 2 reversed.
    inst = Instrument(
        name="mini", version="0", source="",
        scale_min=1, scale_max=5,
        low_label="", high_label="", midpoint_label="",
        axes={"EI": {"pole_low": "E", "pole_high": "I"}},
        type_order=["EI"],
        items=[
            Item(id=1, axis="EI", keyed=1, left="E", right="I"),
            Item(id=2, axis="EI", keyed=-1, left="I", right="E"),
        ],
    )
    # score 5 on item1 -> oriented 5 (toward I); score 5 on item2 reversed ->
    # oriented (1+5)-5 = 1 (toward E). Sum = 6, midpoint = 2*3 = 6 -> tie -> E.
    a = score_axis(inst, "EI", {1: 5, 2: 5})
    assert a.raw_sum == 6
    assert a.letter == "E"
    # Now both point toward I: item1=5 (oriented 5), item2=1 (oriented 5). Sum=10>6 -> I.
    a2 = score_axis(inst, "EI", {1: 5, 2: 1})
    assert a2.raw_sum == 10
    assert a2.letter == "I"
    assert a2.pct_high == 100.0
