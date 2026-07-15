"""Robustly parse the model's JSON reply into {item_id: score}.

Models occasionally wrap JSON in markdown fences or add a stray sentence. The
parser tolerates those, but is strict about the payload: every expected id must
be present exactly once, and every score must be an integer within the scale.
Anything else raises ParseError, which the orchestrator treats as a retryable
failure.
"""

from __future__ import annotations

import json
import re


class ParseError(ValueError):
    """Raised when a reply cannot be parsed into a complete, valid answer set."""


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _extract_json_blob(text: str) -> str:
    """Pull the most plausible JSON object/array out of arbitrary model text."""
    text = text.strip()
    if not text:
        raise ParseError("Empty response.")

    # 1) If the whole thing already parses, use it verbatim.
    if _is_json(text):
        return text

    # 2) Prefer a fenced code block if present.
    fence = _FENCE_RE.search(text)
    if fence:
        candidate = fence.group(1).strip()
        if candidate:
            return candidate

    # 3) Otherwise take the widest object {...} or array [...] span present.
    candidates = []
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        end = text.rfind(close_ch)
        if start != -1 and end != -1 and end > start:
            candidates.append(text[start : end + 1])
    for cand in candidates:
        if _is_json(cand):
            return cand
    if candidates:
        return candidates[0]

    raise ParseError("No JSON object found in response.")


def _is_json(s: str) -> bool:
    try:
        json.loads(s)
        return True
    except json.JSONDecodeError:
        return False


def parse_answers(
    text: str, expected_ids: list[int], scale_min: int, scale_max: int
) -> dict[int, int]:
    """Parse `text` into {id: score}, validating completeness and range."""
    blob = _extract_json_blob(text)
    try:
        data = json.loads(blob)
    except json.JSONDecodeError as exc:
        raise ParseError(f"Invalid JSON: {exc}") from exc

    # Accept either {"answers": [...]} or a bare list, or a flat mapping
    # {"1": 3, "2": 5, ...} as a fallback.
    answers = _normalize(data)

    parsed: dict[int, int] = {}
    for entry in answers:
        try:
            item_id = int(entry["id"])
            score = entry["score"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ParseError(f"Malformed answer entry: {entry!r}") from exc
        if isinstance(score, bool):  # bool is an int subclass; reject it
            raise ParseError(f"Non-numeric score for id {item_id}: {score!r}")
        if isinstance(score, float):
            if not score.is_integer():
                raise ParseError(f"Non-integer score for id {item_id}: {score!r}")
            score = int(score)
        if not isinstance(score, int):
            raise ParseError(f"Non-integer score for id {item_id}: {score!r}")
        if item_id in parsed:
            raise ParseError(f"Duplicate answer for id {item_id}.")
        if not (scale_min <= score <= scale_max):
            raise ParseError(
                f"Score {score} for id {item_id} out of range [{scale_min},{scale_max}]."
            )
        parsed[item_id] = score

    expected = set(expected_ids)
    got = set(parsed)
    if got != expected:
        missing = sorted(expected - got)
        extra = sorted(got - expected)
        raise ParseError(
            f"Answer id mismatch. missing={missing} extra={extra}."
        )
    return parsed


def _normalize(data) -> list[dict]:
    if isinstance(data, dict) and "answers" in data:
        answers = data["answers"]
    elif isinstance(data, list):
        answers = data
    elif isinstance(data, dict):
        # Flat mapping {"1": 3, ...} -> list of {id, score}.
        answers = [{"id": k, "score": v} for k, v in data.items()]
    else:
        raise ParseError("Unexpected JSON top-level structure.")
    if not isinstance(answers, list):
        raise ParseError("'answers' must be a list.")
    return answers
