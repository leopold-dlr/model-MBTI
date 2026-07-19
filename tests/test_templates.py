from pathlib import Path

import pytest

from src.instrument import load_instrument
from src.prompting import templates

INSTRUMENT = Path(__file__).resolve().parent.parent / "config" / "instrument" / "oejts_32.yaml"


def test_flip_map_is_deterministic():
    inst = load_instrument(INSTRUMENT)
    a = templates.flip_map(inst, seed=123)
    b = templates.flip_map(inst, seed=123)
    assert a == b


def test_flip_map_differs_by_seed():
    inst = load_instrument(INSTRUMENT)
    a = templates.flip_map(inst, seed=1)
    b = templates.flip_map(inst, seed=2)
    assert a != b


def test_flip_map_flips_half_of_each_axis():
    inst = load_instrument(INSTRUMENT)
    flips = templates.flip_map(inst, seed=7)
    for axis in inst.type_order:
        ids = [it.id for it in inst.items_for_axis(axis)]
        flipped = [i for i in ids if flips.get(i) == -1]
        assert len(flipped) == len(ids) // 2


def test_flip_map_swaps_displayed_anchors():
    inst = load_instrument(INSTRUMENT)
    flips = templates.flip_map(inst, seed=7)
    order = templates.ordered_items(inst, seed=7, randomize=False)
    user = templates.build_user_prompt(inst, order, flips=flips)
    flipped_item = next(it for it in inst.items if flips.get(it.id) == -1)
    unflipped_item = next(it for it in inst.items if flips.get(it.id) != -1)
    assert f'LEFT = "{flipped_item.right}"' in user
    assert f'RIGHT = "{flipped_item.left}"' in user
    assert f'LEFT = "{unflipped_item.left}"' in user
    assert f'RIGHT = "{unflipped_item.right}"' in user


def test_keyed_overrides_from_flips_inverts_canonical_keyed():
    inst = load_instrument(INSTRUMENT)
    flips = templates.flip_map(inst, seed=7)
    overrides = templates.keyed_overrides_from_flips(inst, flips)
    for it in inst.items:
        if flips.get(it.id) == -1:
            assert overrides[it.id] == -it.keyed
        else:
            assert it.id not in overrides


def test_build_prompt_unknown_variant_raises():
    inst = load_instrument(INSTRUMENT)
    with pytest.raises(KeyError):
        templates.build_prompt(inst, seed=1, randomize=False, prompt_variant="nonexistent")


def test_build_prompt_variants_share_json_contract():
    inst = load_instrument(INSTRUMENT)
    for variant in templates.SYSTEM_PROMPTS:
        system, user, order = templates.build_prompt(
            inst, seed=1, randomize=False, prompt_variant=variant
        )
        assert "STRICT JSON" in system
        assert len(order) == len(inst.items)
