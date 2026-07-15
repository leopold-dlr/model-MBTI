import pytest

from src.prompting.parser import ParseError, parse_answers

IDS = [1, 2, 3]


def test_parses_answers_wrapper():
    txt = '{"answers":[{"id":1,"score":5},{"id":2,"score":3},{"id":3,"score":1}]}'
    assert parse_answers(txt, IDS, 1, 5) == {1: 5, 2: 3, 3: 1}


def test_strips_markdown_fence_and_prose():
    txt = 'Sure!\n```json\n{"answers":[{"id":1,"score":2},{"id":2,"score":2},{"id":3,"score":4}]}\n```\nDone.'
    assert parse_answers(txt, IDS, 1, 5) == {1: 2, 2: 2, 3: 4}


def test_bare_list():
    txt = '[{"id":1,"score":1},{"id":2,"score":1},{"id":3,"score":1}]'
    assert parse_answers(txt, IDS, 1, 5) == {1: 1, 2: 1, 3: 1}


def test_flat_mapping_fallback():
    txt = '{"1": 4, "2": 4, "3": 4}'
    assert parse_answers(txt, IDS, 1, 5) == {1: 4, 2: 4, 3: 4}


def test_float_integer_scores_ok():
    txt = '{"answers":[{"id":1,"score":5.0},{"id":2,"score":3},{"id":3,"score":1}]}'
    assert parse_answers(txt, IDS, 1, 5) == {1: 5, 2: 3, 3: 1}


def test_missing_id_raises():
    txt = '{"answers":[{"id":1,"score":5},{"id":2,"score":3}]}'
    with pytest.raises(ParseError):
        parse_answers(txt, IDS, 1, 5)


def test_extra_id_raises():
    txt = '{"answers":[{"id":1,"score":5},{"id":2,"score":3},{"id":3,"score":1},{"id":4,"score":1}]}'
    with pytest.raises(ParseError):
        parse_answers(txt, IDS, 1, 5)


def test_out_of_range_raises():
    txt = '{"answers":[{"id":1,"score":9},{"id":2,"score":3},{"id":3,"score":1}]}'
    with pytest.raises(ParseError):
        parse_answers(txt, IDS, 1, 5)


def test_non_integer_score_raises():
    txt = '{"answers":[{"id":1,"score":2.5},{"id":2,"score":3},{"id":3,"score":1}]}'
    with pytest.raises(ParseError):
        parse_answers(txt, IDS, 1, 5)


def test_refusal_text_raises():
    txt = "As an AI, I don't have a personality, so I can't answer this."
    with pytest.raises(ParseError):
        parse_answers(txt, IDS, 1, 5)


def test_duplicate_id_raises():
    txt = '{"answers":[{"id":1,"score":5},{"id":1,"score":3},{"id":3,"score":1}]}'
    with pytest.raises(ParseError):
        parse_answers(txt, IDS, 1, 5)
