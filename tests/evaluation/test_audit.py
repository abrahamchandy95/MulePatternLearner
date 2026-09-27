"""The final population audit and its weighted review-budget metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from mule_pattern_learner.batching import assemble
from mule_pattern_learner.contract.graph_schema import ContextKey
from mule_pattern_learner.evaluation.audit import evaluate_final_population, evaluate_weighted
from mule_pattern_learner.metrics import evaluate
from mule_pattern_learner.testing.builders import (
    CUTOFFS,
    base_config,
    checkpoint,
    hub_registry,
    prepared_dataset,
)
from mule_pattern_learner.testing.fake_graph import FakeSource, ScoringExecutor

TOP_KEYS = [f"{kind}_at_{pct}pct" for pct in (1, 5, 10) for kind in ("precision", "recall")]
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


def test_final_population_audit_scores_through_the_dataset_clock_and_hubs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = base_config(max_rejected_root_fraction=0.1)
    dataset, _, accounts = prepared_dataset(tmp_path / "dataset", config, monkeypatch)
    model = checkpoint(tmp_path / "model.pt", config, dataset / "manifest.json")
    test_accounts = accounts[accounts.split == "test"]
    truth = test_accounts[["account_id"]].assign(is_mule=(np.arange(len(test_accounts)) % 4 == 0))
    truth["is_mule"] = truth.is_mule.astype(int)
    source = FakeSource(config, reject=frozenset({test_accounts.account_id.iloc[1]}))
    seen: list[int] = []
    real = assemble.make_live_batch

    def record(store: Any, roots: list[ContextKey], **kwargs: Any) -> dict[str, torch.Tensor]:
        seen.extend(k.cutoff_seq for k in roots)
        return real(store, roots, **kwargs)

    monkeypatch.setattr(assemble, "make_live_batch", record)

    class Truth:
        def read(self) -> pd.DataFrame:
            return truth

    result = evaluate_final_population(
        model,
        Truth(),
        tmp_path / "final.json",
        executor=ScoringExecutor(test_accounts),
        contexts=source,
        hubs=hub_registry(),
    )
    assert set(seen) == {CUTOFFS["2025-01-01"]}
    assert result["rejected_accounts"] == 1 and result["rejected_negatives"] == 1
    assert result["rejected"] == 1 and result["rejected_roots_by_status"] == {"missing_entity": 1}
    assert result["metrics"]["sample_accounts"] == len(test_accounts) - 1
    assert result["metrics"]["evaluation_cohort"].endswith("_minus_rejected_negatives")
    for pct in (1, 5, 10):
        assert 0 <= result["metrics"][f"recall_at_{pct}pct"] <= 1
        assert 0 <= result["metrics"][f"precision_at_{pct}pct"] <= 1
    assert pd.read_parquet(tmp_path / "final.parquet").score.dtype == np.float64
    assert (tmp_path / "final.rejected.txt").read_text().split() == [
        test_accounts.account_id.iloc[1]
    ]


@pytest.mark.parametrize(("limit", "rejected_index"), [(1.0, 0), (0.0, 1)])
def test_final_population_audit_fails_on_censored_rejections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, limit: float, rejected_index: int
) -> None:
    config = base_config(max_rejected_root_fraction=limit)
    dataset, _, accounts = prepared_dataset(tmp_path / "dataset", config, monkeypatch)
    model = checkpoint(tmp_path / "model.pt", config, dataset / "manifest.json")
    test_accounts = accounts[accounts.split == "test"]
    truth = test_accounts[["account_id"]].assign(
        is_mule=(np.arange(len(test_accounts)) % 4 == 0).astype(int)
    )

    class Truth:
        def read(self) -> pd.DataFrame:
            return truth

    # Index 0 is a test positive (always fatal); index 1 a negative (fatal at limit 0).
    source = FakeSource(config, reject=frozenset({test_accounts.account_id.iloc[rejected_index]}))
    match = "1 test positives" if rejected_index == 0 else "0 test positives"
    with pytest.raises(ValueError, match=match):
        evaluate_final_population(
            model,
            Truth(),
            tmp_path / "final.json",
            executor=ScoringExecutor(test_accounts),
            contexts=source,
            hubs=hub_registry(),
        )
    assert not list(tmp_path.glob("final*"))


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
