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

SYSTEM_PROMPT = (
    "You are completing a short self-report questionnaire.\n\n"
    "Answer each item based on your own actual behavioral and reasoning "
    "tendencies as this model -- not as a character, persona, or hypothetical "
    "person you are asked to imagine. Answer as yourself. There are no right or "
    "wrong answers; report your genuine default tendencies.\n\n"
    "You must respond with STRICT JSON only -- no prose, no explanation, no "
    "markdown fences. Do not skip any item."
)

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


def build_user_prompt(inst: Instrument, item_order: list[Item]) -> str:
    lines = [
        f"Below are {len(item_order)} items. Each presents a LEFT statement and a "
        "RIGHT statement.",
        "",
        _scale_legend(inst),
        "",
        "Items:",
    ]
    for it in item_order:
        lines.append(f'  id {it.id}: LEFT = "{it.left}"  |  RIGHT = "{it.right}"')

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
    inst: Instrument, seed: int, randomize: bool, retry: bool = False
) -> tuple[str, str, list[int]]:
    """Return (system_prompt, user_prompt, item_id_order)."""
    order = ordered_items(inst, seed=seed, randomize=randomize)
    system = SYSTEM_PROMPT + (RETRY_REMINDER if retry else "")
    user = build_user_prompt(inst, order)
    return system, user, [it.id for it in order]
