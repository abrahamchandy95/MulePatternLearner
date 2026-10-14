"""The answer key as read_answer_key returns it, a page at a time: evaluation reads it, and
no other code does."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..tigergraph.queries import parse_row

MULE_LAYERS = frozenset({"", "first_layer", "onward_layer"})
MULE_FLOWS = frozenset({"", "pass_through", "collector", "cash_out", "received_only"})


@dataclass(frozen=True)
class AnswerKeyRow:
    """An account's truth. is_mule is 1, 0, or -1 when unknown; the times are 0 for none."""

    account_id: str
    is_mule: int
    mule_ring_id: int
    mule_layer: str
    mule_flow: str
    mule_onset_ms: int
    mule_last_ms: int
    discovered_ms: int


def parse_answer_key(page: Sequence[Mapping[str, Any]]) -> list[AnswerKeyRow]:
    """A page of read_answer_key (as tigergraph.executor.account_pages yields it)."""
    rows = [
        parse_row(AnswerKeyRow, row, signed=frozenset({"is_mule", "mule_ring_id"})) for row in page
    ]
    for row in rows:
        if (
            row.is_mule not in (-1, 0, 1)
            or row.mule_layer not in MULE_LAYERS
            or row.mule_flow not in MULE_FLOWS
        ):
            raise ValueError(f"read_answer_key returned an invalid truth for {row.account_id}")
    return rows
