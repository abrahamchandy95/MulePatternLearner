"""When the audited mules were active, from the reveal's inputs, beside the run's ranking."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from mule_pattern_learner.artifacts import DIAGNOSTIC_TABLES
from mule_pattern_learner.contract.clock import timestamp
from mule_pattern_learner.diagnostics.activity_timing import (
    DAY_MS,
    activity_timing,
    fraud_inflows,
    timing,
)

CUTOFF = "2024-10-01"
AT = timestamp(CUTOFF)


def test_a_mule_is_timed_by_its_last_fraud_inflow_before_the_cutoff() -> None:
    assert timing([], AT) == "none"
    assert timing([AT, AT + DAY_MS], AT) == "not_yet"  # the cutoff's own instant is after
    assert timing([AT - 30 * DAY_MS], AT) == "recent"
    assert timing([AT - 30 * DAY_MS - 1], AT) == "earlier"
    assert timing([AT - 90 * DAY_MS], AT) == "earlier"
    assert timing([AT - 200 * DAY_MS], AT) == "long_before"
    # The last inflow before the cutoff decides, whatever came after it.
    assert timing([AT - 200 * DAY_MS, AT - 5 * DAY_MS, AT + DAY_MS], AT) == "recent"


def mule(account: str, *days_before: int) -> dict[str, Any]:
    inflows = [f"{i}:{AT - day * DAY_MS}" for i, day in enumerate(days_before)]
    attributes = {"M.id": account, "M.@part": 2, "M.@key": 1, "M.@inflows": inflows}
    return {"v_id": account, "attributes": {**attributes, "M.first_seen_ts_ms": 0}}


# The reveal's inputs: the mules' fraud inflows, days before the cutoff (negative: after).
INPUTS = [
    {
        "M": [
            mule("recent", 10),
            mule("earlier", 60),
            mule("long_before", 200),
            mule("not_yet", -10),
            mule("none"),
            mule("revealed", 5),
        ]
    },
    {"zelle_links": []},
    {"payment_links": []},
]


def audit_sample() -> pd.DataFrame:
    """Six mules (weight 1, one revealed) and ten non-mules standing for ten accounts each."""
    mules = {
        "recent": 0.99,
        "revealed": 0.98,
        "earlier": 0.96,
        "long_before": 0.45,
        "not_yet": 0.32,
        "none": 0.05,
    }
    others = [0.95, 0.7, 0.6, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15]
    return pd.DataFrame(
        {
            "account_id": [*mules, *(f"n{i}" for i in range(len(others)))],
            "is_mule": [1] * len(mules) + [0] * len(others),
            "inclusion_probability": [1.0] * len(mules) + [0.1] * len(others),
            "score": [*mules.values(), *others],
            "revealed": [name == "revealed" for name in mules] + [False] * len(others),
            "ring_id": -1,
        }
    )


def test_each_timing_counts_its_mules_and_where_the_ranking_put_them() -> None:
    table = activity_timing({"validation": audit_sample()}, {"validation": CUTOFF}, INPUTS)
    assert tuple(table.columns) == DIAGNOSTIC_TABLES["activity_timing"]

    def value(subset: str, kind: str, metric: str) -> float:
        rows = table[(table.subset == subset) & (table.timing == kind) & (table.metric == metric)]
        assert len(rows) == 1, (subset, kind, metric)
        return float(rows.value.iloc[0])

    for kind in ("recent", "earlier", "long_before", "not_yet", "none"):
        assert value("hidden", kind, "mules") == 1
    # The hidden ranking holds 105 accounts, the revealed mule removed. Recent leads, at
    # 1/105 of it, and earlier follows at 2/105 (in the top 5%, not the top 1%);
    # long_before has four non-mules, standing for 40 accounts, above it.
    assert [value("hidden", "recent", f"in_top_{b}") for b in ("1pct", "5pct", "10pct")] == [
        1,
        1,
        1,
    ]
    assert [value("hidden", "earlier", f"in_top_{b}") for b in ("1pct", "5pct", "10pct")] == [
        0,
        1,
        1,
    ]
    assert value("hidden", "long_before", "in_top_10pct") == 0
    assert value("hidden", "long_before", "median_population_rank") == 40
    assert value("hidden", "recent", "roc_auc") == pytest.approx(1.0)
    assert value("hidden", "none", "roc_auc") == pytest.approx(0.0)
    # The revealed mule is ranked with the hidden ones removed.
    assert value("revealed", "recent", "mules") == 1
    assert value("revealed", "recent", "in_top_1pct") == 1
    assert value("revealed", "earlier", "mules") == 0
    # A timing without mules has its count alone.
    empty = table[(table.subset == "revealed") & (table.timing == "earlier")]
    assert empty.metric.tolist() == ["mules"]


def test_mules_missing_from_the_reveal_inputs_are_refused() -> None:
    inputs = [{"M": [row for row in INPUTS[0]["M"] if row["v_id"] != "none"]}]
    with pytest.raises(ValueError, match="1 validation mules are not among the reveal's inputs"):
        activity_timing({"validation": audit_sample()}, {"validation": CUTOFF}, inputs)
    with pytest.raises(ValueError, match="no mules result"):
        fraud_inflows([{"zelle_links": []}])
    assert fraud_inflows(INPUTS)["earlier"] == [AT - 60 * DAY_MS]
