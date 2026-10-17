"""read_answer_key: every internal account's truth, a page at a time by strictly increasing id
(docs/design.md, "read_answer_key"), for evaluation only.

Only evaluation/ and testing/ import this module: the registry names the query and its version
but does not import it, and an import contract in pyproject.toml refuses it to every other
package (the Labels guard). The fetch executor refuses to run the query.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ...contract import ACCOUNT_TYPES
from . import READ_ANSWER_KEY, VERSIONS
from .rows import after, page_size, paged, response_head

# An internal account's is_mule: 1 for a mule, 0 for any other (the label convention, by
# which -1 is exactly an external account's).
INTERNAL_LABELS = (0, 1)


@dataclass(frozen=True)
class AccountTruth:
    """An internal account's truth: whether it is a mule, the ring of its first landing
    payment or the ring that recruited it (-1 for none), its activity period, its opening (for
    the group of accounts opened within 90 days of t), its product and whether it has an
    ownership link at all."""

    account_id: str
    is_mule: int
    mule_ring_id: int
    mule_onset_ms: int
    mule_last_ms: int
    opened_ms: int
    account_type: str
    owned: bool


def params(after_id: str, page: int) -> dict[str, Any]:
    """The parameters of one page: the internal accounts after after_id, at most page of
    them."""
    return {"after_id": after(after_id), "page": page_size(page)}


def parse(rows: Sequence[object], after_id: str, page: int) -> tuple[AccountTruth, ...]:
    """The accounts of the page that params(after_id, page) asked for. An is_mule other than 0
    or 1, a ring below -1 or an account_type outside the contract's is refused."""
    response_head(rows, READ_ANSWER_KEY, VERSIONS[READ_ANSWER_KEY])
    truths = paged(
        AccountTruth, rows, READ_ANSWER_KEY, after_id, page, signed=frozenset({"mule_ring_id"})
    )
    for truth in truths:
        if truth.is_mule not in INTERNAL_LABELS or truth.mule_ring_id < -1:
            raise ValueError(
                f"{READ_ANSWER_KEY} answered {truth.account_id} with is_mule {truth.is_mule} "
                f"and ring {truth.mule_ring_id}"
            )
        if truth.account_type not in ACCOUNT_TYPES:
            raise ValueError(
                f"{READ_ANSWER_KEY} answered {truth.account_id} of account_type "
                f"{truth.account_type!r}, not one of {ACCOUNT_TYPES}"
            )
    return truths
