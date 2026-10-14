"""The graph's revealed positives for training."""

from __future__ import annotations

import pandas as pd

from mule_pattern_learner.tigergraph.labels import TigerGraphObservedLabelReader

POPULATION = pd.DataFrame(
    {
        "account_id": ["a", "b", "c"],
        "split": ["train", "validation", "train"],
        "observed_positive": [True, True, False],
        "known_from_ms": [1_000, 2_000, 0],
    }
)


def test_the_population_rows_give_the_revealed_positives_and_their_clocks() -> None:
    labels = TigerGraphObservedLabelReader().read(POPULATION)
    assert labels.known_positive.tolist() == [True, True, False]
    assert labels.known_from_ms.tolist() == [1_000, 2_000, 0]
    # Only revealed train positives are training labels.
    assert labels.pu_label.tolist() == [1, 0, 0]
