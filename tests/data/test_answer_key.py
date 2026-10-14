"""The answer key's pages, as read_answer_key prints them, parsed for evaluation, and the
answer key of the reference graph's world read into a frame."""

from __future__ import annotations

from typing import Any

import pytest

from mule_pattern_learner.data.answer_key import (
    AnswerKeyRow,
    parse_answer_key,
    read_answer_key,
)
from mule_pattern_learner.testing.world import HOUR_MS, standard_world

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


def test_the_answer_key_of_a_world_is_one_typed_row_per_account() -> None:
    graph = standard_world().graph()
    truth = read_answer_key(graph)
    assert list(truth.account_id) == sorted(graph.accounts)
    assert list(truth.columns) == [
        "account_id",
        "is_mule",
        "mule_ring_id",
        "mule_layer",
        "mule_flow",
        "mule_onset_ms",
        "mule_last_ms",
        "discovered_ms",
    ]
    assert {str(dtype) for dtype in truth.dtypes} == {"str", "int64"}
    mia = truth.set_index("account_id").loc["A_MIA"]
    assert tuple(mia) == (
        1,
        0,
        "first_layer",
        "pass_through",
        T - 60 * DAY,
        T - 10 * DAY + 2 * HOUR_MS,
        T - 5 * DAY,
    )
    assert set(truth.account_id[truth.is_mule == 1]) == {"A_MIA", "A_MAX", "A_ROY", "A_SUE"}
    assert list(truth.account_id[truth.is_mule == -1]) == ["X_CASH"]
