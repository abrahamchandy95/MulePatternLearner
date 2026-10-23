"""The clock probe of mule audit (docs/design.md, the Clock guard).

t is always a Monday at 00:00 UTC, so a duration measured to t gives its event's clock time
back: a since_last_out of 16 to 17 hours is a Sunday 07:xx outflow. The probe asks whether the
clock times of every duration measured to t predict is_mule, which a model could learn as the
simulator's clock habits rather than mule behaviour.

For each account-Monday of the audit sample (evaluation.sample: every active true mule, and a
keyed sample of non-mule account-Mondays), the events of its durations measured to t, in seven
families: the age of its own payments (B4's payments it sent or received), since_last_in and
since_last_out (B3), since_start and since_last_seen of its owner-enrolment edges (B5), and
since_first_use and since_last_use of its seen edges (B5). Of each family, its EVENTS newest
events (the shortest durations), each as the event time t - duration reduced modulo a minute,
an hour, a day and a week, and its day of the month (UTC): 7 x 8 x 5 = 280 numbers, -1 where a
family has fewer events. Durations that are absent give no event.

XGBoost with its default settings learns is_mule from them in two folds, each cluster of the
ring bootstrap in one fold by a hash of its name (a ring with all its account-Mondays, a
ringless mule and a non-mule account each its own), so that no ring is learned in one fold and
scored in the other. The out-of-fold scores give a ROC AUC with a ring-resampled interval
(evaluation.uncertainty), and the probe passes when the interval's upper end is below the
owner's 0.55 (audit.clock_probe_upper). A probe with too few mules has a wide interval, or none,
and fails, so low power never passes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import xgboost

from ..comparisons.rankers import require_one_openmp_thread
from ..config import AuditConfig
from ..subgraph.build import ExampleRows
from .uncertainty import Clusters, Ranked, interval

# The families of durations measured to t, and the events kept of each.
FAMILIES = (
    "age",
    "since_last_in",
    "since_last_out",
    "since_start",
    "since_last_seen",
    "since_first_use",
    "since_last_use",
)
EVENTS = 8
# The clock residues of each event: modulo a minute, an hour, a day and a week, in ms; then its
# day of the month.
MODULI = (60_000, 3_600_000, 86_400_000, 604_800_000)
PER_EVENT = len(MODULI) + 1
WIDTH = len(FAMILIES) * EVENTS * PER_EVENT
# The value of an event a family lacks.
MISSING = -1.0
# The folds of the probe.
FOLDS = 2


def durations(rows: ExampleRows) -> dict[str, list[int]]:
    """The durations measured to t of an example's scored account, by family: those present."""
    scored = rows.scored
    activity = rows.activity[scored]
    owner = [edge for edge in rows.identity.owner_enrolled if edge.account_id == scored]
    seen = [edge for edge in rows.identity.seen if edge.account_id == scored]
    return {
        "age": [
            payment.age_ms
            for payment in rows.payments.payments
            if scored in (payment.sender_id, payment.receiver_id)
        ],
        "since_last_in": [activity.since_last_in_ms] if activity.since_last_in_present else [],
        "since_last_out": [activity.since_last_out_ms] if activity.since_last_out_present else [],
        "since_start": [edge.since_start_ms for edge in owner if edge.since_start_present],
        "since_last_seen": [
            edge.since_last_seen_ms for edge in owner if edge.since_last_seen_present
        ],
        "since_first_use": [
            edge.since_first_use_ms for edge in seen if edge.since_first_use_present
        ],
        "since_last_use": [edge.since_last_use_ms for edge in seen],
    }


def clock_values(t_ms: int, found: dict[str, list[int]]) -> NDArray[np.float64]:
    """An example's 280 numbers: for each family its EVENTS newest events, each as its time's
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


def folds(keys: Sequence[str]) -> NDArray[np.int64]:
    """The fold of each row: one of FOLDS by a hash of its cluster's name, so that a cluster's
    rows share their fold."""
    found: dict[str, int] = {}
    for key in set(keys):
        digest = hashlib.blake2b(f"clock-probe|{key}".encode(), digest_size=8).digest()
        found[key] = int.from_bytes(digest, "big") % FOLDS
    return np.array([found[key] for key in keys], dtype=np.int64)


@dataclass(frozen=True)
class Probe:
    """What the probe found: its ROC AUC, the ring-resampled interval, whether it passed, and
    the true-mule account-Mondays and clusters it read."""

    roc_auc: float
    interval: tuple[float, float] | None
    passed: bool
    mule_rows: int
    clusters: int

    def frame(self) -> pd.DataFrame:
        """The probe as a one-row report table."""
        low, high = self.interval if self.interval is not None else (np.nan, np.nan)
        return pd.DataFrame(
            {
                "roc_auc": [self.roc_auc],
                "low": [low],
                "high": [high],
                "passed": [self.passed],
                "mule_rows": [self.mule_rows],
                "clusters": [self.clusters],
            }
        )


def out_of_fold(
    values: NDArray[np.float64], positive: NDArray[np.bool_], fold: NDArray[np.int64], seed: int
) -> NDArray[np.float64] | None:
    """Each row's score from XGBoost with its default settings trained on the other fold; None
    when a fold lacks a class to learn from."""
    require_one_openmp_thread()
    scores = np.zeros(len(values))
    for held in range(FOLDS):
        train = fold != held
        if len(set(positive[train].tolist())) < 2 or not (fold == held).any():
            return None
        model = xgboost.XGBClassifier(random_state=seed)
        model.fit(values[train], positive[train].astype(np.int64))
        margins = model.predict(values[~train], output_margin=True)
        scores[~train] = np.asarray(margins, dtype=np.float64)
    return scores


def probe(
    values: NDArray[np.float64],
    positive: NDArray[np.bool_],
    weight: NDArray[np.float64],
    keys: Sequence[str],
    audit: AuditConfig,
    seed: int,
) -> Probe:
    """The probe on the audit sample's clock values (a row of WIDTH per account-Monday), its
    classes, sample weights and cluster names."""
    if values.shape != (len(positive), WIDTH):
        raise ValueError(f"The clock values are {values.shape}, not ({len(positive)}, {WIDTH})")
    clusters = Clusters.of(keys, positive)
    mules = int(positive.sum())
    scores = out_of_fold(values, positive, folds(keys), seed)
    if scores is None:
        return Probe(float("nan"), None, False, mules, len(clusters))
    ranked = Ranked(scores, positive)
    found = interval(
        ranked.roc_auc,
        clusters,
        weight,
        replicates=audit.replicates,
        level=audit.interval,
        seed=seed,
    )
    passed = found is not None and found[1] < audit.clock_probe_upper
    return Probe(ranked.roc_auc(weight), found, passed, mules, len(clusters))
