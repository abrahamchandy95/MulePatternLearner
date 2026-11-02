"""What mule prepare reports of a world (docs/design.md, "Training world" and "Checks on every
world"), from pu_label alone, and the refusal of a selection world with too few identified
active mules.

Per week: the scorable accounts, the positives (identified mules active and scorable then),
the identified mules dropped (scorable then and not active: examples in their other weeks),
and for a training world the unlabelled pool, the draw, and the sums of each label's weights
and their ratio. If identified mules cluster in some months, seasonal spending could act as a
clock; the identification horizon gate holds the cause flat, and this table shows the effect.
For the world: the identified mules, the identified active mules (positive in some week),
the identified active share (positives over scorable accounts, averaged over weeks), which
sets the prior grid, and the identified mules never scored in a week of their activity.

The per-week table is written beside the examples as report.parquet.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd
import pyarrow as pa

from ..runtime.console import plural
from .examples import POSITIVE, UNLABELLED, Draw, Tally
from .labels import Labels, Unscored

REPORT_FILE = "report.parquet"
REPORT_SCHEMA = pa.schema(
    [
        ("t_ms", pa.int64()),
        ("scorable", pa.int64()),
        ("positives", pa.int64()),
        ("dropped", pa.int64()),
        ("pool", pa.int64()),
        ("unlabelled", pa.int64()),
        ("positive_weight", pa.float64()),
        ("unlabelled_weight", pa.float64()),
        ("ratio", pa.float64()),
    ]
)


def identified_active_share(weeks: pd.DataFrame) -> float:
    """Identified active mules over scorable accounts, averaged over the weeks of a per-week
    table (REPORT_SCHEMA's columns) that have a scorable account. mule train reads it from the
    table mule prepare wrote, to set the prior grid."""
    held = weeks[weeks["scorable"] > 0]
    if held.empty:
        return 0.0
    return float((held["positives"] / held["scorable"]).mean())


@dataclass(frozen=True)
class Report:
    """The per-week table, in REPORT_SCHEMA's columns, and the world's counts."""

    weeks: pd.DataFrame
    identified: int
    identified_active: int
    unscored: Unscored

    @property
    def identified_active_share(self) -> float:
        """The share of the per-week table (identified_active_share)."""
        return identified_active_share(self.weeks)


def build_report(
    tallies: Sequence[Tally],
    labels: Labels,
    unscored: Unscored,
    draws: Sequence[Draw] = (),
    training: pd.DataFrame | None = None,
) -> Report:
    """The report of a world's weeks; draws and training only for a training world."""
    pools = {draw.t_ms: draw for draw in draws}
    weights: dict[tuple[int, int], float] = {}
    if training is not None:
        for t_ms, label, weight in zip(
            training["t_ms"], training["label"], training["weight"], strict=True
        ):
            key = (int(t_ms), int(label))
            weights[key] = weights.get(key, 0.0) + float(weight)
    rows: list[tuple[int, int, int, int, int, int, float, float, float]] = []
    for tally in tallies:
        draw = pools.get(tally.t_ms)
        positive = weights.get((tally.t_ms, POSITIVE), 0.0)
        unlabelled = weights.get((tally.t_ms, UNLABELLED), 0.0)
        rows.append(
            (
                tally.t_ms,
                tally.scorable,
                len(tally.positives),
                tally.dropped,
                draw.pool if draw else 0,
                len(draw.drawn) if draw else 0,
                positive,
                unlabelled,
                positive / unlabelled if unlabelled else 0.0,
            )
        )
    frame = pd.DataFrame(rows, columns=REPORT_SCHEMA.names)
    frame = frame.astype({name: "int64" for name in REPORT_SCHEMA.names[:6]})
    active_mules = {account for tally in tallies for account in tally.positives}
    return Report(frame, len(labels.mules), len(active_mules), unscored)


def refuse_selection(report: Report, minimum: int) -> None:
    """Refuse a world that selects with fewer than minimum identified active mules."""
    if report.identified_active < minimum:
        raise ValueError(
            f"The selection world has {plural(report.identified_active, 'identified active mule')}"
            f", fewer than the {minimum} selection needs"
        )


def lines(report: Report) -> list[str]:
    """The report as a person reads it."""
    frame = report.weeks
    found = [
        f"{plural(report.identified, 'identified mule')}, "
        f"{plural(report.identified_active, 'identified active mule')}; identified active "
        f"share {report.identified_active_share:.6f}",
        f"{plural(int(frame['positives'].sum()), 'positive')} and "
        f"{plural(int(frame['unlabelled'].sum()), 'unlabelled example')} over "
        f"{plural(len(frame), 'week')}",
    ]
    if report.unscored.no_active_week:
        found.append(
            f"{plural(len(report.unscored.no_active_week), 'identified mule')} active in no "
            f"scored week: {', '.join(report.unscored.no_active_week)}"
        )
    if report.unscored.never_scorable:
        found.append(
            f"{plural(len(report.unscored.never_scorable), 'identified mule')} never scorable "
            f"in a week of its activity: {', '.join(report.unscored.never_scorable)}"
        )
    return found
