"""The nnPU positive weight on a small synthetic problem."""

from __future__ import annotations

from mule_pattern_learner.artifacts import DIAGNOSTIC_TABLES
from mule_pattern_learner.diagnostics.nnpu_simulation import (
    COLLAPSE,
    PRIOR,
    WEIGHTS,
    Problem,
    nnpu_simulation,
)

# A problem small enough for a test: a tenth of the accounts and a few short epochs.
SMALL = Problem(marginal=2_000, test_positives=30, test_negatives=30_000, steps=20, epochs=3)


def test_every_weight_and_seed_records_its_kept_epoch_and_whether_it_collapsed() -> None:
    table = nnpu_simulation(weights=(PRIOR, 1 - PRIOR), seeds=(1, 2), problem=SMALL)
    assert tuple(table.columns) == DIAGNOSTIC_TABLES["nnpu_simulation"]
    assert sorted(table.positive_weight.unique()) == [PRIOR, 1 - PRIOR]
    wide = table.pivot_table(index=["positive_weight", "seed"], columns="metric", values="value")
    assert len(wide) == 4
    assert ((wide.test_roc_auc >= 0) & (wide.test_roc_auc <= 1)).all()
    assert (wide.kept_epoch <= wide.stopped_epoch).all() and (wide.stopped_epoch <= 3).all()
    assert (wide.collapsed == (wide.labelled_positive_mean_score < COLLAPSE)).all()
    # The textbook weight scores its own positives far lower than the balanced one.
    prior, balanced = (wide.xs(w).labelled_positive_mean_score.mean() for w in (PRIOR, 1 - PRIOR))
    assert prior < balanced
    # The runs are seeded, so the table is the same every time.
    again = nnpu_simulation(weights=(PRIOR, 1 - PRIOR), seeds=(1, 2), problem=SMALL)
    assert again.equals(table)
    assert WEIGHTS[0] == PRIOR and WEIGHTS[-1] == 1 - PRIOR
