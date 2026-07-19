"""Deterministic scoring of a single answer set into a 4-letter type.

Fully network-free and unit-tested (see tests/test_scorer.py). Given a mapping
{item_id: score} and an Instrument, it produces per-axis letters + preference
percentages and the concatenated type string.

Method (per axis):
  * Orient each item toward its pole_high letter:
        oriented = score               if keyed == +1
        oriented = (min + max) - score if keyed == -1   (reverse-keyed)
  * axis_sum = sum(oriented) over the axis's items.
  * midpoint = n_items * (min+max)/2.
        axis_sum >  midpoint -> pole_high letter
        axis_sum <= midpoint -> pole_low  letter  (ties break to pole_low)
  * Preference toward pole_high, as a percentage of the sum's range:
        pct_high = 100 * (axis_sum - n*min) / (n*(max-min))
        pct_low  = 100 - pct_high
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..instrument import Instrument


@dataclass
class AxisScore:
    axis: str
    letter: str            # the dominant letter for this axis
    pole_low: str
    pole_high: str
    raw_sum: float         # oriented sum (toward pole_high)
    n_items: int
    pct_low: float         # preference % for pole_low letter
    pct_high: float        # preference % for pole_high letter

    @property
    def letter_pct(self) -> float:
        """Preference percentage of the dominant (reported) letter."""
        return self.pct_high if self.letter == self.pole_high else self.pct_low

    def to_dict(self) -> dict:
        return {
            "axis": self.axis,
            "letter": self.letter,
            "pole_low": self.pole_low,
            "pole_high": self.pole_high,
            "raw_sum": self.raw_sum,
            "n_items": self.n_items,
            "pct_low": round(self.pct_low, 2),
            "pct_high": round(self.pct_high, 2),
            "letter_pct": round(self.letter_pct, 2),
        }


@dataclass
class TypeResult:
    type: str
    axes: dict[str, AxisScore] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"type": self.type, "axes": {k: v.to_dict() for k, v in self.axes.items()}}


def _oriented(score: int, keyed: int, scale_min: int, scale_max: int) -> int:
    if keyed == 1:
        return score
    return (scale_min + scale_max) - score


def score_axis(
    inst: Instrument,
    axis: str,
    answers: dict[int, int],
    keyed_overrides: dict[int, int] | None = None,
) -> AxisScore:
    """Score one axis.

    ``keyed_overrides`` maps item id -> effective keyed (+1/-1), overriding the
    instrument's canonical ``keyed`` for that item. Used when the item's
    left/right anchors were swapped in the prompt actually sent for this run
    (see ``prompting.templates.flip_map``), so scoring stays consistent with
    what the model actually saw.
    """
    items = inst.items_for_axis(axis)
    n = len(items)
    if n == 0:
        raise ValueError(f"No items for axis '{axis}'.")
    overrides = keyed_overrides or {}
    raw_sum = sum(
        _oriented(
            answers[it.id], overrides.get(it.id, it.keyed), inst.scale_min, inst.scale_max
        )
        for it in items
    )
    pole_low = inst.axes[axis]["pole_low"]
    pole_high = inst.axes[axis]["pole_high"]

    min_sum = n * inst.scale_min
    span = n * (inst.scale_max - inst.scale_min)
    pct_high = 100.0 * (raw_sum - min_sum) / span if span else 50.0
    pct_low = 100.0 - pct_high

    midpoint = n * inst.scale_midpoint
    letter = pole_high if raw_sum > midpoint else pole_low

    return AxisScore(
        axis=axis,
        letter=letter,
        pole_low=pole_low,
        pole_high=pole_high,
        raw_sum=raw_sum,
        n_items=n,
        pct_low=pct_low,
        pct_high=pct_high,
    )


def score_answers(
    inst: Instrument,
    answers: dict[int, int],
    keyed_overrides: dict[int, int] | None = None,
) -> TypeResult:
    """Score a complete answer set. Raises KeyError if an item is unanswered."""
    missing = set(inst.item_ids()) - set(answers)
    if missing:
        raise KeyError(f"Missing answers for item ids: {sorted(missing)}")

    axes: dict[str, AxisScore] = {}
    letters: list[str] = []
    for axis in inst.type_order:
        axis_score = score_axis(inst, axis, answers, keyed_overrides=keyed_overrides)
        axes[axis] = axis_score
        letters.append(axis_score.letter)
    return TypeResult(type="".join(letters), axes=axes)
