"""The label reveal replayed over salts on its inputs, offline."""

from __future__ import annotations

from mule_pattern_learner.artifacts import DIAGNOSTIC_TABLES
from mule_pattern_learner.config import DEFAULT_CONFIG
from mule_pattern_learner.diagnostics.reveal_spread import SALTS, reveal_spread
from mule_pattern_learner.reference.label_reveal import counts_by_split, plan
from mule_pattern_learner.testing.builders import reveal_inputs
from mule_pattern_learner.tigergraph.reveal import reveal_parameters


def test_each_salt_replays_the_mirror_of_the_reveal_per_split() -> None:
    params = reveal_parameters(DEFAULT_CONFIG.scope, DEFAULT_CONFIG.dataset.dates, apply=False)
    table = reveal_spread(reveal_inputs(), params, salts=range(3))
    assert tuple(table.columns) == DIAGNOSTIC_TABLES["reveal_spread"]
    assert sorted(table.salt.unique()) == [0, 1, 2]
    mules = table[table.metric == "mules"].groupby("split").value.first()
    assert mules.to_dict() == {"train": 2, "validation": 1, "test": 1}
    for salt in range(3):
        expected = plan(reveal_inputs(), {**params, "salt": salt})
        rows = table[table.salt == salt]
        for metric in ("eligible", "revealed"):
            found = rows[rows.metric == metric].set_index("split").value
            counts = counts_by_split(expected, metric)
            assert found.to_dict() == {
                "train": counts[1],
                "validation": counts[2],
                "test": counts[3],
            }
    # The replayed salts hold the configured one.
    assert DEFAULT_CONFIG.scope.reveal_salt in SALTS
