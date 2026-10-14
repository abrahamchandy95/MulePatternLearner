"""The customer hold-out (docs/design.md, "Splits and hidden labels"): a quarter of the
customers, chosen by a seeded hash of the party id, written on the graph by write_split."""

from __future__ import annotations

from collections.abc import Iterable
import hashlib

from ..tigergraph.executor import QueryExecutor
from ..tigergraph.queries import WRITE_SPLIT, SplitWritten, parse_split, split_params

# The share of customers held out.
HELD_OUT_SHARE = 0.25


def held_out(party_id: str, seed: int) -> bool:
    """Whether the customer is held out: the 64-bit blake2b hash of its id, keyed by seed,
    falls in the lowest HELD_OUT_SHARE of its range. Ids that differ by one character hash
    to unrelated values, so neighbouring ids are held out independently."""
    digest = hashlib.blake2b(party_id.encode(), digest_size=8, key=str(seed).encode()).digest()
    return int.from_bytes(digest) < HELD_OUT_SHARE * 2**64


def splits(party_ids: Iterable[str], seed: int) -> dict[str, int]:
    """Each party's split: 1 when it is held out, else 0."""
    return {party_id: int(held_out(party_id, seed)) for party_id in party_ids}


def write_holdout(executor: QueryExecutor, party_ids: Iterable[str], seed: int) -> SplitWritten:
    """Writes the split of every party, and of the accounts it owns, in one write_split.

    party_ids must name every party: one write_split never wrote keeps split -1 and is no
    customer in any phase. A single request, because an account owned by several parties
    takes the largest split among those of its request.
    """
    chosen = splits(party_ids, seed)
    written = parse_split(executor.run(WRITE_SPLIT, split_params(chosen), attempts=1))
    if written.parties != len(chosen):
        raise ValueError(f"write_split wrote {written.parties} parties, not {len(chosen)}")
    return written
