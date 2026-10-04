"""The learning curve, on a synthetic feature table."""

from __future__ import annotations

from mule_pattern_learner.artifacts import DIAGNOSTIC_TABLES
from mule_pattern_learner.diagnostics.learning_curve import CURVE_METRICS, learning_curve
from mule_pattern_learner.testing.builders import FEATURE_SAMPLE, feature_frame


def test_random_draws_up_to_every_train_mule_then_the_revealed_labels_and_the_run() -> None:
    frame = feature_frame()
    audits = {
        "test": {
            "hidden_metrics": {"average_precision": 0.05, "roc_auc": 0.88},
            "metrics": {"average_precision": 0.13, "roc_auc": 0.93},
        }
    }
    table = learning_curve(frame, counts=(10, 20, 80), repeats=3, audits=audits)
    assert tuple(table.columns) == DIAGNOSTIC_TABLES["learning_curve"]
    mules = FEATURE_SAMPLE["train"][0]
    random = table[table.labels == "random"]
    # 80 is more than the 40 train mules, so it becomes all of them, drawn once.
    assert sorted(random.mules.unique()) == [10, 20, mules]
    draws = random.groupby(["model", "mules"]).repeat.nunique()
    assert draws.to_dict() == {
        (kind, k): (1 if k == mules else 3) for kind in ("hgb", "lr") for k in (10, 20, mules)
    }
    assert set(random.metric) == set(CURVE_METRICS)
    revealed = table[(table.labels == "revealed") & (table.model != "model")]
    train = frame[(frame.split == "train") & ~frame.rejected]
    count = int((train.is_mule.eq(1) & train.revealed).sum())
    assert set(revealed.mules) == {count} and set(revealed.model) == {"lr", "hgb"}
    run = table[table.model == "model"]
    # The run's audit of the hidden mules, then of every mule.
    assert run[["mules", "split", "metric", "value"]].to_numpy().tolist() == [
        [count, "test", "hidden_average_precision", 0.05],
        [count, "test", "hidden_roc_auc", 0.88],
        [count, "test", "average_precision", 0.13],
        [count, "test", "roc_auc", 0.93],
    ]
    # More labels rank better on average.
    auc = random[(random.metric == "roc_auc") & (random.split == "test")]
    means = auc.groupby("mules").value.mean()
    assert means[mules] >= means[10]
    # The draws are fixed by the seed.
    again = learning_curve(frame, counts=(10, 20, 80), repeats=3, audits=audits)
    assert again.equals(table)
