"""Build the system + user prompt for a one-shot administration of the test.

Design constraints (from the project plan, section 4):

* No role-play. The model answers as *itself*, reporting its own default
  behavioral/reasoning tendencies -- not a character it is asked to imagine.
* Closed questions only. Each item is a semantic-differential 1..5 rating
  between two anchors; no free text, no justification.
* One-shot. All items go out in a single request; a single strict-JSON reply
  is expected back.
"""

from __future__ import annotations

import random

from ..instrument import Instrument, Item

# Named system-prompt framings, for an (opt-in) ablation of prompt wording --
# see run_settings.yaml `prompt_variants`. Sharing the same *semantics* (answer
# as yourself, no role-play, strict JSON) but different phrasing lets you check
# whether results are an artifact of one specific wording. Only "default" is
# used unless the config enables more, so the cost of a run is unchanged
# unless you opt in.
SYSTEM_PROMPTS: dict[str, str] = {
    "default": (
        "You are completing a short self-report questionnaire.\n\n"
        "Answer each item based on your own actual behavioral and reasoning "
        "tendencies as this model -- not as a character, persona, or hypothetical "
        "person you are asked to imagine. Answer as yourself. There are no right or "
        "wrong answers; report your genuine default tendencies.\n\n"
        "You must respond with STRICT JSON only -- no prose, no explanation, no "
        "markdown fences. Do not skip any item."
    ),
    "as_ai_explicit": (
        "You are an AI language model completing a short self-report "
        "questionnaire about your own default behavioral and reasoning "
        "tendencies -- the tendencies your outputs actually exhibit, not a "
        "human personality you are asked to simulate or a character you are "
        "asked to play. Answer as the AI system you are. There are no right "
        "or wrong answers.\n\n"
        "You must respond with STRICT JSON only -- no prose, no explanation, "
        "no markdown fences. Do not skip any item."
    ),
    "no_meta_mention": (
        "Complete the following self-report questionnaire about your own "
        "default tendencies. Answer as yourself, not as a character or "
        "hypothetical person. There are no right or wrong answers; report "
        "your genuine default tendencies.\n\n"
        "Respond with STRICT JSON only -- no prose, no explanation, no "
        "markdown fences. Do not skip any item."
    ),
}
DEFAULT_PROMPT_VARIANT = "default"

# Appended to the system prompt on a retry, to firmly re-state the contract.
RETRY_REMINDER = (
    "\n\nIMPORTANT: your previous reply could not be parsed. Reply with STRICT "
    "JSON ONLY, matching the exact schema requested, with an integer score for "
    "every item id and nothing else. Answer as yourself -- do not decline, and "
    "do not add any commentary."
)


def _scale_legend(inst: Instrument) -> str:
    return (
        f"Rate each item on an integer scale from {inst.scale_min} to {inst.scale_max}:\n"
        f"  {inst.scale_min} = {inst.low_label}\n"
        f"  {inst.scale_midpoint:g} = {inst.midpoint_label}\n"
        f"  {inst.scale_max} = {inst.high_label}\n"
        f"For each item, {inst.scale_min} leans fully toward the LEFT statement "
        f"and {inst.scale_max} leans fully toward the RIGHT statement."
    )


def ordered_items(inst: Instrument, seed: int, randomize: bool) -> list[Item]:
    items = list(inst.items)
    if randomize:
        rng = random.Random(seed)
        rng.shuffle(items)
    return items


def flip_map(inst: Instrument, seed: int) -> dict[int, int]:
    """Deterministically choose ~half of each axis's items to counterbalance
    left/right anchor polarity for this run.

    Every item in the instrument is keyed the same direction (pole_high always
    on the RIGHT anchor -- see oejts_32.yaml), so a model with any systematic
    left/right or acquiescence bias would land on the same letter regardless
    of its actual tendencies. Flipping which side each pole is displayed on
    for a deterministic ~half of items per axis (and inverting their
    *effective* keyed sign to match, so scoring stays correct) neutralizes
    that confound without touching the instrument's canonical data file.

    Returns {item_id: -1} for flipped items (effective keyed inverted); items
    not present in the map keep their canonical keyed value (+1 semantics:
    "use inst value").
    """
    rng = random.Random(seed ^ 0x5A5A5A5A)
    flips: dict[int, int] = {}
    for axis in inst.type_order:
        ids = [it.id for it in inst.items_for_axis(axis)]
        rng.shuffle(ids)
        half = len(ids) // 2
        for item_id in ids[:half]:
            flips[item_id] = -1
    return flips


def keyed_overrides_from_flips(inst: Instrument, flips: dict[int, int]) -> dict[int, int]:
    """Turn a flip map (item_id -> -1 for flipped) into effective keyed values
    for the scorer, honoring the instrument's own canonical keyed sign too."""
    overrides: dict[int, int] = {}
    for it in inst.items:
        if flips.get(it.id) == -1:
            overrides[it.id] = -it.keyed
    return overrides


def build_user_prompt(
    inst: Instrument, item_order: list[Item], flips: dict[int, int] | None = None
) -> str:
    flips = flips or {}
    lines = [
        f"Below are {len(item_order)} items. Each presents a LEFT statement and a "
        "RIGHT statement.",
        "",
        _scale_legend(inst),
        "",
        "Items:",
    ]
    for it in item_order:
        left, right = it.left, it.right
        if flips.get(it.id) == -1:
            left, right = right, left
        lines.append(f'  id {it.id}: LEFT = "{left}"  |  RIGHT = "{right}"')

    ids = [it.id for it in item_order]
    example_id = ids[0]
    lines += [
        "",
        "Respond with a single JSON object of this exact shape:",
        '  {"answers": [{"id": <item id>, "score": <integer '
        f"{inst.scale_min}-{inst.scale_max}>}}, ...]}}",
        f"Include exactly one entry for every id ({len(ids)} entries total). "
        f'For example, an entry looks like {{"id": {example_id}, "score": 3}}.',
        "Output the JSON object and nothing else.",
    ]
    return "\n".join(lines)


def build_prompt(
    inst: Instrument,
    seed: int,
    randomize: bool,
    retry: bool = False,
    flips: dict[int, int] | None = None,
    prompt_variant: str = DEFAULT_PROMPT_VARIANT,
) -> tuple[str, str, list[int]]:
    """Return (system_prompt, user_prompt, item_id_order)."""
    order = ordered_items(inst, seed=seed, randomize=randomize)
    base = SYSTEM_PROMPTS.get(prompt_variant)
    if base is None:
        raise KeyError(
            f"Unknown prompt_variant '{prompt_variant}'. Known: {', '.join(SYSTEM_PROMPTS)}."
        )
    system = base + (RETRY_REMINDER if retry else "")
    user = build_user_prompt(inst, order, flips=flips)
    return system, user, [it.id for it in order]
