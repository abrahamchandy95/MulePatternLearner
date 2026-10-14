"""The answer key's pages, as read_answer_key prints them, parsed for evaluation."""

from __future__ import annotations

from typing import Any

import pytest

from mule_pattern_learner.evaluation.answer_key import AnswerKeyRow, parse_answer_key

DAY = 86_400_000
T = 1_719_792_000_000

PAGE: list[dict[str, Any]] = [
    {
        "account_id": "A1",
        "is_mule": 1,
        "mule_ring_id": 4,
        "mule_layer": "first_layer",
        "mule_flow": "pass_through",
        "mule_onset_ms": T - 40 * DAY,
        "mule_last_ms": T - 2 * DAY,
        "discovered_ms": T + 10 * DAY,
    },
    {
        "account_id": "A2",
        "is_mule": 0,
        "mule_ring_id": -1,
        "mule_layer": "",
        "mule_flow": "",
        "mule_onset_ms": 0,
        "mule_last_ms": 0,
        "discovered_ms": 0,
    },
    {
        "account_id": "X9",
        "is_mule": -1,
        "mule_ring_id": -1,
        "mule_layer": "",
        "mule_flow": "",
        "mule_onset_ms": 0,
        "mule_last_ms": 0,
        "discovered_ms": 0,
    },
]


def test_a_page_parses_with_unknown_truth_kept_unknown() -> None:
    rows = parse_answer_key(PAGE)
    assert rows[0] == AnswerKeyRow(
        "A1", 1, 4, "first_layer", "pass_through", T - 40 * DAY, T - 2 * DAY, T + 10 * DAY
    )
    assert [row.is_mule for row in rows] == [1, 0, -1]


@pytest.mark.parametrize(
    "change",
    [
        {"is_mule": 2},
        {"mule_layer": "third_layer"},
        {"mule_flow": "laundered"},
        {"mule_onset_ms": -1},
        {"discovered_ms": "soon"},
    ],
)
def test_an_invalid_truth_is_refused(change: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        parse_answer_key([{**PAGE[0], **change}])
