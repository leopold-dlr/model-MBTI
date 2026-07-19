"""Load and validate a psychometric instrument from YAML.

The instrument is *data*, not code: swapping OEJTS for another questionnaire is
a config change, provided the new file follows the same schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Item:
    id: int
    axis: str
    keyed: int  # +1 => high score credits pole_high; -1 => reversed
    left: str
    right: str


@dataclass(frozen=True)
class Instrument:
    name: str
    version: str
    source: str
    scale_min: int
    scale_max: int
    low_label: str
    high_label: str
    midpoint_label: str
    axes: dict[str, dict[str, str]]  # axis -> {pole_low, pole_high}
    type_order: list[str]
    items: list[Item]

    @property
    def scale_midpoint(self) -> float:
        return (self.scale_min + self.scale_max) / 2

    def items_for_axis(self, axis: str) -> list[Item]:
        return [it for it in self.items if it.axis == axis]

    def item_ids(self) -> list[int]:
        return [it.id for it in self.items]


def load_instrument(path: str | Path) -> Instrument:
    data: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    scale = data["scale"]
    items = [
        Item(
            id=int(it["id"]),
            axis=str(it["axis"]),
            keyed=int(it.get("keyed", 1)),
            left=str(it["left"]),
            right=str(it["right"]),
        )
        for it in data["items"]
    ]

    inst = Instrument(
        name=str(data["name"]),
        version=str(data.get("version", "")),
        source=str(data.get("source", "")),
        scale_min=int(scale["min"]),
        scale_max=int(scale["max"]),
        low_label=str(scale.get("low_label", "")),
        high_label=str(scale.get("high_label", "")),
        midpoint_label=str(scale.get("midpoint_label", "")),
        axes={k: dict(v) for k, v in data["axes"].items()},
        type_order=list(data["type_order"]),
        items=items,
    )
    _validate(inst)
    return inst


def _validate(inst: Instrument) -> None:
    if not inst.items:
        raise ValueError(
            f"Instrument '{inst.name}' has no items -- refusing to load. "
            "(If this is a scaffold file, populate its `items:` list first.)"
        )
    ids = [it.id for it in inst.items]
    if len(ids) != len(set(ids)):
        raise ValueError("Instrument has duplicate item ids.")
    for it in inst.items:
        if it.axis not in inst.axes:
            raise ValueError(f"Item {it.id} references unknown axis '{it.axis}'.")
        if it.keyed not in (-1, 1):
            raise ValueError(f"Item {it.id} has invalid keyed value {it.keyed} (must be +1/-1).")
    for axis in inst.type_order:
        if axis not in inst.axes:
            raise ValueError(f"type_order references unknown axis '{axis}'.")
        pole = inst.axes[axis]
        if "pole_low" not in pole or "pole_high" not in pole:
            raise ValueError(f"Axis '{axis}' must define pole_low and pole_high.")
    if inst.scale_min >= inst.scale_max:
        raise ValueError("scale min must be < max.")
