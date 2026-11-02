"""The labels (docs/design.md, "Labels"): the identified mules of read_labels, the only training
signal and the only reader of pu_label, read once per world.

- read_labels refuses a mule without 0 < mule_onset_ms <= mule_last_ms, and so does its
  parser, so mule prepare refuses such a world.
- A mule is active at a scoring time t when mule_onset_ms < t and t <= mule_last_ms + W(t),
  with the window W(t) the longer of 7 days and the time since the previous scoring time (7
  days when there is none): its first ring payment is in view, and it made a ring payment in the
  7 days before t or since the previous scoring time. A payment exactly at the previous
  scoring time counts at t, since the events at that time were not in view at it. So every ring
  payment from 7 days before the first scoring time on counts at the first scoring time after
  it, a mule is active at consecutive scoring times, and on scoring times a week apart the
  window is 7 days. window_start is the earliest last payment that keeps a mule active at t. The
  scoring times of the world are passed in, from the calendar (CalendarConfig.scoring_times),
  never recomputed here.
- The label identity is the hash of read_labels' rows; it keys the training examples, so
  examples of other labels are never read.
- mule prepare reports every identified mule never in the scored population at a scoring time
  at which it is active: those active at no scoring time, and those active at some but scorable
  at none of them.

Nothing here reads the truth: is_mule and the answer key are evaluation's.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..tigergraph.executor import QueryExecutor
from ..tigergraph.queries import READ_LABELS, read_labels
from ..tigergraph.queries.read_labels import IdentifiedMule
from ..tigergraph.queries.rows import DAY_MS, PAGE_LIMIT
from ..tigergraph.world import canonical_hash

# The shortest window that reaches back from t: a week.
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


def window_start(t_ms: int, scoring_times: Sequence[int]) -> int:
    """The earliest last payment that keeps a mule active at t: t - W(t), with W(t) the longer
    of 7 days and the time since the previous scoring time (the last of scoring_times, in
    increasing order, before t; 7 days when none is). That is the smaller of t - 7 days and the
    previous scoring time, so it never decreases along the scoring times."""
    start = t_ms - ACTIVE_AFTER_MS
    before = bisect_left(scoring_times, t_ms)
    return min(start, scoring_times[before - 1]) if before else start


def window_starts(t_ms: NDArray[np.int64], scoring_times: Sequence[int]) -> NDArray[np.int64]:
    """window_start for every t of an array, as a vectorised form of it."""
    start = t_ms - ACTIVE_AFTER_MS
    times = np.asarray(scoring_times, dtype=np.int64)
    if not len(times):
        return start
    before = np.searchsorted(times, t_ms, side="left")
    return np.where(before > 0, np.minimum(start, times[np.maximum(before - 1, 0)]), start)


def active_at(onset_ms: int, last_ms: int, t_ms: int, scoring_times: Sequence[int]) -> bool:
    """Whether a mule active from onset_ms to last_ms is active at t, as the module says: its
    onset strictly before t, so an onset exactly at t is not in view, and its last payment no
    earlier than window_start, so one exactly 7 days before t, or exactly at the previous scoring
    time, still counts. scoring_times are the world's, in increasing order."""
    return onset_ms < t_ms and last_ms >= window_start(t_ms, scoring_times)


def active_rows(
    onset_ms: NDArray[np.int64],
    last_ms: NDArray[np.int64],
    t_ms: NDArray[np.int64],
    scoring_times: Sequence[int],
) -> NDArray[np.bool_]:
    """active_at for every row of aligned arrays, as a vectorised form of it."""
    return (onset_ms < t_ms) & (last_ms >= window_starts(t_ms, scoring_times))


def active(mule: IdentifiedMule, t_ms: int, scoring_times: Sequence[int]) -> bool:
    """Whether the identified mule is active at the scoring time t (active_at)."""
    return active_at(mule.mule_onset_ms, mule.mule_last_ms, t_ms, scoring_times)


def active_mules(
    mules: Sequence[IdentifiedMule], t_ms: int, scoring_times: Sequence[int]
) -> list[IdentifiedMule]:
    """The mules active at the scoring time t."""
    return [mule for mule in mules if active(mule, t_ms, scoring_times)]


@dataclass(frozen=True)
class Unscored:
    """The identified mules never in the scored population at a scoring time of their activity:
    those active at no scoring time, and those active at some but scorable at none."""

    no_active_week: tuple[str, ...]
    never_scorable: tuple[str, ...]


def unscored(
    mules: Sequence[IdentifiedMule], scoring_times: Sequence[int], positives: Mapping[str, int]
) -> Unscored:
    """The identified mules with no positive scoring time, positives giving each mule's count;
    scoring_times are the world's, in increasing order."""
    idle: list[str] = []
    unscorable: list[str] = []
    for mule in mules:
        if positives.get(mule.account_id, 0):
            continue
        if any(active(mule, t_ms, scoring_times) for t_ms in scoring_times):
            unscorable.append(mule.account_id)
        else:
            idle.append(mule.account_id)
    return Unscored(tuple(sorted(idle)), tuple(sorted(unscorable)))
