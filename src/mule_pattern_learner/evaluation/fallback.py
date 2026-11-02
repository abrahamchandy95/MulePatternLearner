"""The test-role fallback list (docs/design.md, "If whole-world scoring is too slow").

When config scores the test world by its fallback (data.scoring "fallback"), mule audit builds
the list while the world is loaded, since it depends on the truth and never on a model: on
every scored week, every true mule scorable then that is active or at an early-warning
time, weighing 1, and a uniform keyed sample of every other scorable account (data.draw's
fallback sample, the same hash as the training draw), each included with chance 1 over
data.fallback_weight and weighing that many, so that no sampled account stands for more than
about 10. The sample is drawn from every other scorable account, true mules in their other
weeks included, at the same rate.

The list goes to the evaluation store, and its rows are fetched through the fetch code
(subgraph.fetch), which never imports evaluation. The cache stores rows only, never why an
account was chosen, and one account's subgraph and score never depend on its batch, so the
list cannot change a feature or a score; training, selection and scoring read only the keys
they computed themselves (the Keys chosen by their reader guard).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from ..config import DataConfig
from ..data.draw import sampled
from .answer_key import ACTIVE, EARLY_WARNING, AnswerKey, status

# What a true mule is in the weeks the list holds it in whatever the sample.
KEPT = frozenset({ACTIVE, EARLY_WARNING})


def fallback_keys(
    key: AnswerKey,
    populations: Mapping[int, Sequence[str]],
    data: DataConfig,
    scoring_times: Sequence[int],
) -> pd.DataFrame:
    """The fallback list: (t_ms, account_id, weight), by week and account. scoring_times are
    the world's, which the active state reads (answer_key.status)."""
    rows: list[tuple[int, str, float]] = []
    for t_ms in sorted(populations):
        kept: list[str] = []
        rest: list[str] = []
        for account in populations[t_ms]:
            truth = key.by_id.get(account)
            if truth is not None and status(truth, t_ms, scoring_times) in KEPT:
                kept.append(account)
            else:
                rest.append(account)
        drawn = sampled(data.draw_key, rest, t_ms, data.fallback_weight)
        rows += [(t_ms, account, 1.0) for account in kept]
        rows += [(t_ms, account, float(data.fallback_weight)) for account in drawn]
    found = pd.DataFrame(rows, columns=["t_ms", "account_id", "weight"]).astype(
        {"t_ms": "int64", "account_id": "str", "weight": "float64"}
    )
    return found.sort_values(["t_ms", "account_id"]).reset_index(drop=True)


def by_week(keys: pd.DataFrame) -> dict[int, list[str]]:
    """The accounts of keys (t_ms, account_id) in each week, in their order."""
    found: dict[int, list[str]] = {}
    for t_ms, account in zip(keys["t_ms"].tolist(), keys["account_id"].tolist(), strict=True):
        found.setdefault(int(t_ms), []).append(str(account))
    return found
