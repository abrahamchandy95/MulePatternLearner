"""The labels (docs/design.md, "Labels"): the identified mules of read_labels, the only training
signal and the only reader of pu_label, read once per world.

- read_labels refuses a mule without 0 < mule_onset_ms <= mule_last_ms, and so does its
  parser, so mule prepare refuses such a world.
- A mule is active on a Monday t when mule_onset_ms < t <= mule_last_ms + 7 days: its first
  ring payment is in view, and the week after its last payment gives it at least one Monday.
- The label identity is the hash of read_labels' rows; it keys the training examples, so
  examples of other labels are never read.
- mule prepare reports every identified mule never in the scored population on a Monday of
  its activity: those whose activity covers no scored Monday, and those active on scored
  Mondays but scorable on none of them.

Nothing here reads the truth: is_mule and the answer key are evaluation's.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

from ..tigergraph.executor import QueryExecutor
from ..tigergraph.queries import READ_LABELS, read_labels
from ..tigergraph.queries.read_labels import IdentifiedMule
from ..tigergraph.queries.rows import DAY_MS, PAGE_LIMIT
from ..tigergraph.world import canonical_hash

# How long after its last payment a mule stays active: the week to the next Monday.
ACTIVE_AFTER_MS = 7 * DAY_MS


@dataclass(frozen=True)
class Labels:
    """The identified mules, by id, and the identity of read_labels' rows."""

    mules: tuple[IdentifiedMule, ...]
    identity: str

    @property
    def accounts(self) -> frozenset[str]:
        """Every identified mule's account: never an unlabelled example."""
        return frozenset(mule.account_id for mule in self.mules)


def label_identity(mules: Sequence[IdentifiedMule]) -> str:
    """The hash of read_labels' rows, by id: 32 hex digits."""
    rows = sorted([mule.account_id, mule.mule_onset_ms, mule.mule_last_ms] for mule in mules)
    return canonical_hash(rows)


def read_identified(executor: QueryExecutor, page: int = PAGE_LIMIT) -> Labels:
    """Every identified mule, from every page of read_labels, until one comes back short."""
    found: list[IdentifiedMule] = []
    after_id = ""
    while True:
        rows = executor.run(READ_LABELS, read_labels.params(after_id, page))
        mules = read_labels.parse(rows, after_id, page)
        found.extend(mules)
        if len(mules) < page:
            return Labels(tuple(found), label_identity(found))
        after_id = mules[-1].account_id


def active(mule: IdentifiedMule, t_ms: int) -> bool:
    """Whether the mule is active on the Monday t: onset < t <= last + 7 days, so an onset
    exactly at t is not, and a last payment exactly 7 days before t is."""
    return mule.mule_onset_ms < t_ms <= mule.mule_last_ms + ACTIVE_AFTER_MS


def active_mules(mules: Sequence[IdentifiedMule], t_ms: int) -> list[IdentifiedMule]:
    """The mules active on the Monday t."""
    return [mule for mule in mules if active(mule, t_ms)]


@dataclass(frozen=True)
class Unscored:
    """The identified mules never in the scored population on a Monday of their activity:
    those active on no scored Monday, and those active on some but scorable on none."""

    no_active_monday: tuple[str, ...]
    never_scorable: tuple[str, ...]


def unscored(
    mules: Sequence[IdentifiedMule], mondays: Collection[int], positives: Mapping[str, int]
) -> Unscored:
    """The identified mules with no positive Monday, positives giving each mule's count."""
    idle: list[str] = []
    unscorable: list[str] = []
    for mule in mules:
        if positives.get(mule.account_id, 0):
            continue
        if any(active(mule, t_ms) for t_ms in mondays):
            unscorable.append(mule.account_id)
        else:
            idle.append(mule.account_id)
    return Unscored(tuple(sorted(idle)), tuple(sorted(unscorable)))
