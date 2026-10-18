"""What mule prepare reports of a world (docs/design.md, "Training world" and "Checks on every
world"), from pu_label alone, and the refusal of a selection world with too few identified
active mules.

Per Monday: the scorable accounts, the positives (identified mules active and scorable then),
the identified mules dropped (scorable then and not active: examples on their other Mondays),
and for a training world the unlabelled pool, the draw, and the sums of each label's weights
and their ratio. If identified mules cluster in some months, seasonal spending could act as a
clock; the identification horizon gate holds the cause flat, and this table shows the effect.
For the world: the identified mules, the identified active mules (positive on some Monday),
the identified active share (positives over scorable accounts, averaged over Mondays), which
sets the prior grid, and the identified mules never scored on a Monday of their activity.

The per-Monday table is written beside the examples as report.parquet.
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


@dataclass(frozen=True)
class Report:
    """The per-Monday table, in REPORT_SCHEMA's columns, and the world's counts."""

    mondays: pd.DataFrame
    identified: int
    identified_active: int
    unscored: Unscored

    @property
    def identified_active_share(self) -> float:
        """Identified active mules over scorable accounts, averaged over the Mondays with a
        scorable account."""
        held = self.mondays[self.mondays["scorable"] > 0]
        if held.empty:
            return 0.0
        return float((held["positives"] / held["scorable"]).mean())


def build_report(
    tallies: Sequence[Tally],
    labels: Labels,
    unscored: Unscored,
    draws: Sequence[Draw] = (),
    training: pd.DataFrame | None = None,
) -> Report:
    """The report of a world's Mondays; draws and training only for a training world."""
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
    frame = report.mondays
    found = [
        f"{plural(report.identified, 'identified mule')}, "
        f"{plural(report.identified_active, 'identified active mule')}; identified active "
        f"share {report.identified_active_share:.6f}",
        f"{plural(int(frame['positives'].sum()), 'positive')} and "
        f"{plural(int(frame['unlabelled'].sum()), 'unlabelled example')} over "
        f"{plural(len(frame), 'Monday')}",
    ]
    if report.unscored.no_active_monday:
        found.append(
            f"{plural(len(report.unscored.no_active_monday), 'identified mule')} active on no "
            f"scored Monday: {', '.join(report.unscored.no_active_monday)}"
        )
    if report.unscored.never_scorable:
        found.append(
            f"{plural(len(report.unscored.never_scorable), 'identified mule')} never scorable "
            f"on a Monday of its activity: {', '.join(report.unscored.never_scorable)}"
        )
    return found
