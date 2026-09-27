"""Float64 scores and the weighted top-fraction metrics of the final audit."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from mule_pattern_learner.metrics import evaluate
from mule_pattern_learner.model.build import probabilities_from_logits
from mule_pattern_learner.temporal.live.evaluation import evaluate_weighted

TOP_KEYS = [f"{kind}_at_{pct}pct" for pct in (1, 5, 10) for kind in ("precision", "recall")]


def test_float64_probabilities_separate_logits_that_tie_in_float32() -> None:
    logits = torch.tensor([11.0, 11.004, 17.0, 20.0, 23.0], requires_grad=True)
    # The former float32 sigmoid: the first two tie, and the last three are exactly 1.
    single = torch.sigmoid(logits).detach().numpy()
    assert single[0] == single[1] and (single[2:] == 1).all()
    scores = probabilities_from_logits(logits)
    assert scores.dtype == np.float64 and (np.diff(scores) > 0).all() and (scores < 1).all()
    # The sigmoid of each float32 logit, to float64 precision.
    exact = 1 / (1 + np.exp(-logits.detach().numpy().astype(np.float64)))
    assert scores == pytest.approx(exact, rel=1e-15, abs=0)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_float64_probabilities_from_mps_logits() -> None:
    # MPS has no float64, so the logits must reach the CPU before the cast.
    logits = torch.tensor([17.0, 20.0], device="mps")
    scores = probabilities_from_logits(logits)
    assert scores.dtype == np.float64 and scores[0] < scores[1] < 1


def audit_frame() -> pd.DataFrame:
    """50 population accounts: 4 mules, all sampled, and 46 non-mules behind 5 sampled ones.

    Ranked by score, the population accounts and mules found so far are: 01 (1, 1), then
    the tied block of 03 and 02 (4, 2), 04 (5, 3), 05 (9, 3), 06 (10, 4), 07 (26, 4),
    08 (42, 4), 09 (50, 4). The block holds 3 population accounts, one of them a mule.
    """
    rows = [
        ("01", 1, 1.0, 0.99),
        ("03", 1, 1.0, 0.97),
        ("02", 0, 0.5, 0.97),
        ("04", 1, 1.0, 0.90),
        ("05", 0, 0.25, 0.80),
        ("06", 1, 1.0, 0.70),
        ("07", 0, 0.0625, 0.50),
        ("08", 0, 0.0625, 0.30),
        ("09", 0, 0.125, 0.20),
    ]
    return pd.DataFrame(rows, columns=["account_id", "is_mule", "inclusion_probability", "score"])


# Top 1% is 0.5 accounts: half of 01. Top 5% is 2.5: 01 and half of the tied block, so
# half of its one mule. Top 10% is 5: 01 to 04.
EXPECTED = {
    "precision_at_1pct": 0.5 / 0.5,
    "recall_at_1pct": 0.5 / 4,
    "precision_at_5pct": 1.5 / 2.5,
    "recall_at_5pct": 1.5 / 4,
    "precision_at_10pct": 3 / 5,
    "recall_at_10pct": 3 / 4,
}


def test_weighted_top_fractions_count_population_accounts() -> None:
    frame = audit_frame()
    metrics = evaluate_weighted(frame, 0.5)
    assert metrics["estimated_population"] == 50 and metrics["weighted_prevalence"] == 0.08
    assert {k: metrics[k] for k in TOP_KEYS} == pytest.approx(EXPECTED)
    # Neither row order nor account IDs break the tie: either order of 03 and 02 would
    # find 2 or 1 mules in the top 5%, and the block counts their average.
    shuffled = frame.sample(frac=1, random_state=1).reset_index(drop=True)
    renamed = frame.assign(account_id=frame.account_id.replace({"02": "03", "03": "02"}))
    for variant in (shuffled, renamed, frame.drop(columns="account_id")):
        assert {k: evaluate_weighted(variant, 0.5)[k] for k in TOP_KEYS} == pytest.approx(EXPECTED)
    # Untied, the order decides: 03 first holds its mule inside the top 5%.
    untied = frame.assign(score=frame.score.where(frame.account_id != "03", 0.98))
    assert evaluate_weighted(untied, 0.5)["recall_at_5pct"] == pytest.approx(2 / 4)


def test_weighted_top_fractions_rank_scores_beyond_float32_precision() -> None:
    frame = audit_frame()
    # The same ranking squeezed within 1e-9 of 1, where float32 rounds every score to 1.
    near_one = frame.assign(score=1 - 1e-9 * (1 - frame.score))
    assert (near_one.score.astype(np.float32) == 1).all()
    assert {k: evaluate_weighted(near_one, 0.5)[k] for k in TOP_KEYS} == pytest.approx(EXPECTED)


def test_unit_weights_give_the_unweighted_top_fraction_metrics() -> None:
    rng = np.random.default_rng(3)
    # 1, 5 and 10% of 200 accounts are whole numbers: 2, 10 and 20.
    y = (rng.random(200) < 0.1).astype(np.int64)
    score = rng.random(200)
    frame = pd.DataFrame(
        {
            "account_id": [f"A{i:03d}" for i in range(200)],
            "is_mule": y,
            "inclusion_probability": 1.0,
            "score": score,
        }
    )
    weighted = evaluate_weighted(frame, 0.5)
    unweighted = evaluate(y, score, 0.5)
    for key in ("precision_at_1pct", "recall_at_1pct", "precision_at_5pct", "recall_at_5pct"):
        assert weighted[key] == pytest.approx(unweighted[key])
    top = np.argsort(-score)[:20]
    assert weighted["precision_at_10pct"] == pytest.approx(y[top].sum() / 20)
    assert weighted["recall_at_10pct"] == pytest.approx(y[top].sum() / y.sum())


def test_weighted_top_fractions_without_positives_are_zero() -> None:
    frame = audit_frame().assign(is_mule=0)
    metrics = evaluate_weighted(frame, 0.5)
    assert all(metrics[k] == 0 for k in TOP_KEYS) and metrics["average_precision"] is None
