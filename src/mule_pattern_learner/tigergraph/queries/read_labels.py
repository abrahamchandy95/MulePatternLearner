"""read_labels: the identified mules (pu_label 1) and their activity period, a page at a time
by strictly increasing id (docs/design.md, "read_labels"). It is the only training signal."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .rows import after, page_size, paged, response_head


@dataclass(frozen=True)
class IdentifiedMule:
    """An account the bank identified as a mule, and its activity period: its first and last
    ring payment, or for a recruit no ring money reached, its recruitment and its leaving the
    ring."""

    account_id: str
    mule_onset_ms: int
    mule_last_ms: int


def params(after_id: str, page: int) -> dict[str, Any]:
    """The parameters of one page: the identified mules after after_id, at most page of
    them."""
    return {"after_id": after(after_id), "page": page_size(page)}


def parse(rows: Sequence[object], after_id: str, page: int) -> tuple[IdentifiedMule, ...]:
    """The identified mules of the page that params(after_id, page) asked for. A mule without
    0 < mule_onset_ms <= mule_last_ms is refused, as the query refuses it."""
    from . import READ_LABELS, VERSIONS  # the registry imports this module

    response_head(rows, READ_LABELS, VERSIONS[READ_LABELS])
    mules = paged(IdentifiedMule, rows, READ_LABELS, after_id, page)
    for mule in mules:
        if not 0 < mule.mule_onset_ms <= mule.mule_last_ms:
            raise ValueError(
                f"{READ_LABELS} answered {mule.account_id} active from {mule.mule_onset_ms} to "
                f"{mule.mule_last_ms}, not 0 < onset <= last"
            )
    return mules
