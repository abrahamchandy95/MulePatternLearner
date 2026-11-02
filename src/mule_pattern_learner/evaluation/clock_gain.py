"""The clock gain test of mule audit (docs/design.md, the Clock guard).

A duration measured to t is its event's clock time shifted by t. At a fixed t, a Monday at 00:00
UTC, every duration would give its event's clock time back (a since_last_out of 16 to 17 hours is
a Sunday 07:xx outflow), and a model could learn the generated data's clock habits instead of
mule behaviour. Each week is scored at a time of its own inside it (data.calendar,
scoring_times), so a model cannot read an event's clock time from a duration, and what the audit
asks is a property of the data: whether the clock times of events separate mules from non-mules
beyond what plain activity does. The invariance test of the trained model
(evaluation.clock_invariance) checks the model directly.

For each account-week of the audit sample (evaluation.sample: every active true mule, and a
keyed sample of non-mule account-weeks), the events of its durations measured to t, in
seven families (OWN) of two scopes. The scored account's own: the age of its own payments
(B4's payments it sent or received), since_last_in and since_last_out (B3), since_start and
since_last_seen of its owner-enrolment edges (B5), and since_first_use and since_last_use of
its seen edges (B5). And, apart, the same seven of everything else in its subgraph (the families
named other_...), since the model reads them through message passing too: the age of every
other payment in the subgraph (a neighbour's payments reach the score at hop 1 and 2), the
since_last_in and since_last_out of every other account node, the since_start and
since_last_seen of every enrolment edge of a customer (a co-user's among them) and of the
owner-enrolment edges of every other account, and the since_first_use and since_last_use of
the seen edges of every other account. A clock habit visible on a neighbour's payment, a co-user's
enrolment or a hop-1 account's seen edge alone would reach the score and escape a test that
read the scored account's own edges only. Durations that are absent give no event. Two blocks of
numbers are made of them:

- the clock values: of each family, its EVENTS newest events (the shortest durations), each as
  the event time t - duration reduced modulo a minute, an hour, a day and a week, and its day of
  the month (UTC): 14 x 8 x 5 = 560 numbers, -1 where a family has fewer events;
- the controls: for each family the number of its events with a present duration, the number of
  nodes of each type in the example's subgraph (account, customer, device and IP), and each of
  the same newest events' duration floored to whole days, -1 where a family has fewer events:
  14 + 4 + 14 x 8 = 130 numbers. They hold what plain activity says (how many events, how
  recent, how big a subgraph) and nothing of an event's time of day, day of week or day of month.

XGBoost with its default settings learns is_mule out of fold from the controls alone and from the
controls with the clock values, in two folds, each cluster of the ring bootstrap in one fold by a
hash of its name (a ring with all its account-weeks, a ringless mule and a non-mule account each
its own), so that no ring is learned in one fold and scored in the other. It runs in a Python
process of its own that never imports torch (comparisons.process), as every XGBoost fit does. The
gain is the out-of-fold ROC AUC with the clock values minus the one without them. Its interval
comes from resampling the clusters of the ring bootstrap once and computing both ROC AUCs on each
resample (evaluation.uncertainty), so that the two are paired. The test passes only when the
sample holds at least audit.clock_min_mule_clusters (30) mule clusters (the positive clusters of
evaluation.uncertainty.Clusters: each ring, and each ringless mule alone) and the interval's upper
end is below the owner's 0.02 (audit.clock_gain_upper). A few mules give a wide interval, or none,
or a lucky narrow one (3 rings and 10 non-mules passed on the interval alone in 2 of 6 seeds), so
below the minimum the test fails whatever its interval says, and low power never passes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from ..comparisons import process
from ..config import AuditConfig
from ..subgraph.build import ExampleRows, account_nodes
from ..subgraph.layouts import ACCOUNT, CUSTOMER, DEVICE, IP, NODE_TYPES
from ..tigergraph.queries.rows import DAY_MS
from .uncertainty import Clusters, Ranked, interval

# The families of durations measured to t of the scored account's own edges and rows, those of
# everything else in its subgraph under the same names with OTHER before them, and the events
# kept of each.
OWN = (
    "age",
    "since_last_in",
    "since_last_out",
    "since_start",
    "since_last_seen",
    "since_first_use",
    "since_last_use",
)
OTHER = "other_"
FAMILIES = (*OWN, *(f"{OTHER}{name}" for name in OWN))
EVENTS = 8
# The clock residues of each event: modulo a minute, an hour, a day and a week, in ms; then its
# day of the month.
MODULI = (60_000, 3_600_000, 86_400_000, 604_800_000)
PER_EVENT = len(MODULI) + 1
WIDTH = len(FAMILIES) * EVENTS * PER_EVENT
# The numbers of the controls: each family's events, each type of node, each kept event's days.
CONTROLS = len(FAMILIES) + len(NODE_TYPES) + len(FAMILIES) * EVENTS
# The value of an event a family lacks.
MISSING = -1.0
# The folds of the test.
FOLDS = 2


def durations(rows: ExampleRows) -> dict[str, list[int]]:
    """The durations measured to t of an example, by family (FAMILIES): those present, of the
    scored account's own edges and rows, and of every other node and edge of its subgraph."""
    scored = rows.scored
    mine = rows.activity[scored]
    identity = rows.identity
    payments = rows.payments.payments
    others = [row for account, row in rows.activity.items() if account != scored]
    owner = [edge for edge in identity.owner_enrolled if edge.account_id == scored]
    elsewhere = [edge for edge in identity.owner_enrolled if edge.account_id != scored]
    enrolled = [*identity.enrolled, *elsewhere]
    seen = [edge for edge in identity.seen if edge.account_id == scored]
    seen_elsewhere = [edge for edge in identity.seen if edge.account_id != scored]
    return {
        "age": [p.age_ms for p in payments if scored in (p.sender_id, p.receiver_id)],
        "since_last_in": [mine.since_last_in_ms] if mine.since_last_in_present else [],
        "since_last_out": [mine.since_last_out_ms] if mine.since_last_out_present else [],
        "since_start": [edge.since_start_ms for edge in owner if edge.since_start_present],
        "since_last_seen": [
            edge.since_last_seen_ms for edge in owner if edge.since_last_seen_present
        ],
        "since_first_use": [
            edge.since_first_use_ms for edge in seen if edge.since_first_use_present
        ],
        "since_last_use": [edge.since_last_use_ms for edge in seen],
        f"{OTHER}age": [p.age_ms for p in payments if scored not in (p.sender_id, p.receiver_id)],
        f"{OTHER}since_last_in": [
            row.since_last_in_ms for row in others if row.since_last_in_present
        ],
        f"{OTHER}since_last_out": [
            row.since_last_out_ms for row in others if row.since_last_out_present
        ],
        f"{OTHER}since_start": [
            edge.since_start_ms for edge in enrolled if edge.since_start_present
        ],
        f"{OTHER}since_last_seen": [
            edge.since_last_seen_ms for edge in enrolled if edge.since_last_seen_present
        ],
        f"{OTHER}since_first_use": [
            edge.since_first_use_ms for edge in seen_elsewhere if edge.since_first_use_present
        ],
        f"{OTHER}since_last_use": [edge.since_last_use_ms for edge in seen_elsewhere],
    }


def clock_values(t_ms: int, found: dict[str, list[int]]) -> NDArray[np.float64]:
    """An example's WIDTH numbers: for each family its EVENTS newest events, each as its time's
    residues modulo MODULI and its day of the month, MISSING for an event the family lacks."""
    values = np.full((len(FAMILIES), EVENTS, PER_EVENT), MISSING)
    for row, family in enumerate(FAMILIES):
        kept = sorted(found[family])[:EVENTS]
        if not kept:
            continue
        events = t_ms - np.asarray(kept, dtype=np.int64)
        for column, modulus in enumerate(MODULI):
            values[row, : len(kept), column] = events % modulus
        days = events.astype("datetime64[ms]").astype("datetime64[D]")
        months = days.astype("datetime64[M]").astype("datetime64[D]")
        values[row, : len(kept), len(MODULI)] = (days - months).astype(np.int64) + 1
    return values.reshape(-1)


def node_counts(rows: ExampleRows) -> list[int]:
    """The number of nodes of each type (NODE_TYPES) in an example's subgraph: the accounts of
    B4's subgraph and of B5's co-users, and B5's customers, devices and IPs."""
    identity = rows.identity
    found = {
        ACCOUNT: len(account_nodes(rows.payments, identity)),
        CUSTOMER: len(identity.customers),
        DEVICE: len(identity.devices),
        IP: len(identity.ips),
    }
    return [found[kind] for kind in NODE_TYPES]


def control_values(found: dict[str, list[int]], nodes: Sequence[int]) -> NDArray[np.float64]:
    """An example's CONTROLS numbers: the events of each family that have a duration, the nodes
    of each type (node_counts), and for each family its EVENTS newest events' durations floored
    to whole days, MISSING for an event the family lacks."""
    days = np.full((len(FAMILIES), EVENTS), MISSING)
    for row, family in enumerate(FAMILIES):
        kept = sorted(found[family])[:EVENTS]
        days[row, : len(kept)] = np.asarray(kept, dtype=np.int64) // DAY_MS
    counts = [len(found[family]) for family in FAMILIES]
    return np.concatenate([counts, nodes, days.reshape(-1)]).astype(np.float64)


def folds(keys: Sequence[str]) -> NDArray[np.int64]:
    """The fold of each row: one of FOLDS by a hash of its cluster's name, so that a cluster's
    rows share their fold."""
    found: dict[str, int] = {}
    for key in set(keys):
        digest = hashlib.blake2b(f"clock-gain|{key}".encode(), digest_size=8).digest()
        found[key] = int.from_bytes(digest, "big") % FOLDS
    return np.array([found[key] for key in keys], dtype=np.int64)


@dataclass(frozen=True)
class Gain:
    """What the gain test found: the ROC AUC out of fold of the controls alone and with the
    clock values, their difference, its paired ring-resampled interval, whether it passed, and
    the true-mule account-weeks, the clusters and the mule clusters it read."""

    roc_auc_controls: float
    roc_auc_with_clock: float
    gain: float
    interval: tuple[float, float] | None
    passed: bool
    mule_rows: int
    clusters: int
    mule_clusters: int

    def frame(self) -> pd.DataFrame:
        """The test as a one-row report table."""
        low, high = self.interval if self.interval is not None else (np.nan, np.nan)
        return pd.DataFrame(
            {
                "roc_auc_controls": [self.roc_auc_controls],
                "roc_auc_with_clock": [self.roc_auc_with_clock],
                "gain": [self.gain],
                "low": [low],
                "high": [high],
                "passed": [self.passed],
                "mule_rows": [self.mule_rows],
                "clusters": [self.clusters],
                "mule_clusters": [self.mule_clusters],
            }
        )


def out_of_fold(
    values: NDArray[np.float64], positive: NDArray[np.bool_], fold: NDArray[np.int64], seed: int
) -> NDArray[np.float64] | None:
    """Each row's score from XGBoost with its default settings trained on the other fold; None
    when a fold lacks a class to learn from."""
    for held in range(FOLDS):
        train = fold != held
        if len(set(positive[train].tolist())) < 2 or not (fold == held).any():
            return None
    return process.fold_scores(values, positive, fold, seed)


def gain(
    controls: NDArray[np.float64],
    values: NDArray[np.float64],
    positive: NDArray[np.bool_],
    weight: NDArray[np.float64],
    keys: Sequence[str],
    audit: AuditConfig,
    seed: int,
) -> Gain:
    """The gain test on the audit sample's controls (a row of CONTROLS per account-week) and
    clock values (a row of WIDTH), its classes, sample weights and cluster names. It passes only
    with at least audit.clock_min_mule_clusters mule clusters and an interval whose upper end is
    below audit.clock_gain_upper."""
    if controls.shape != (len(positive), CONTROLS):
        raise ValueError(f"The controls are {controls.shape}, not ({len(positive)}, {CONTROLS})")
    if values.shape != (len(positive), WIDTH):
        raise ValueError(f"The clock values are {values.shape}, not ({len(positive)}, {WIDTH})")
    clusters = Clusters.of(keys, positive)
    mules = int(positive.sum())
    mule_clusters = int(clusters.positive.sum())
    fold = folds(keys)
    plain = out_of_fold(controls, positive, fold, seed)
    clocked = out_of_fold(np.hstack([controls, values]), positive, fold, seed)
    if plain is None or clocked is None:
        nan = float("nan")
        return Gain(nan, nan, nan, None, False, mules, len(clusters), mule_clusters)
    without, with_clock = Ranked(plain, positive), Ranked(clocked, positive)

    def difference(draws: NDArray[np.float64]) -> float:
        return with_clock.roc_auc(draws) - without.roc_auc(draws)

    found = interval(
        difference,
        clusters,
        weight,
        replicates=audit.replicates,
        level=audit.interval,
        seed=seed,
    )
    passed = (
        mule_clusters >= audit.clock_min_mule_clusters
        and found is not None
        and found[1] < audit.clock_gain_upper
    )
    return Gain(
        without.roc_auc(weight),
        with_clock.roc_auc(weight),
        difference(weight),
        found,
        passed,
        mules,
        len(clusters),
        mule_clusters,
    )
