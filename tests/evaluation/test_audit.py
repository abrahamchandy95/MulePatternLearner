"""The ground-truth audit of one split of a run's model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from mule_pattern_learner.artifacts import AUDIT_COLUMNS, read_audit_scores, read_json
from mule_pattern_learner.batching import assemble
from mule_pattern_learner.contract.clock import timestamp
from mule_pattern_learner.contract.graph_schema import SPLIT_PHASE, ContextKey
from mule_pattern_learner.evaluation.audit import (
    audit,
    audit_inputs,
    audit_population,
    audit_results,
)
from mule_pattern_learner.paths import RunPaths
from mule_pattern_learner.runtime import console
from mule_pattern_learner.testing.builders import (
    CUTOFFS,
    DATES,
    RUNTIME_CHANGES,
    hub_registry,
    prepared_dataset,
    saved_model,
    unit_config,
)
from mule_pattern_learner.testing.fake_graph import FakeSource, FakeTigerGraph
from mule_pattern_learner.tigergraph.scope import TigerGraphScopeReader


def split_scope(accounts: pd.DataFrame, split: str) -> TigerGraphScopeReader:
    """A scope whose population is these accounts, all in the split's partition."""
    rows = [
        {
            "account_id": account,
            "partition": SPLIT_PHASE[split],
            "first_seen_ts_ms": 1,
            "observed_positive": False,
            "known_from_ms": 0,
        }
        for account in accounts.account_id
    ]
    return TigerGraphScopeReader(FakeTigerGraph(population=rows))


def truth_of(accounts: pd.DataFrame) -> pd.DataFrame:
    """Every fourth account is a mule, and consecutive mules pair up in rings."""
    index = np.arange(len(accounts))
    mule = index % 4 == 0
    return pd.DataFrame(
        {
            "account_id": accounts.account_id.to_numpy(),
            "is_mule": mule.astype(int),
            "ring_id": np.where(mule, index // 8, -1),
            "label_source": np.where(mule, "phantomledger_role;unit;hidden", "phantomledger_role"),
        }
    )


@pytest.mark.parametrize("split", ["validation", "test"])
def test_the_audit_scores_a_split_through_the_dataset_clock_and_hubs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    split: str,
) -> None:
    config = unit_config(RUNTIME_CHANGES, runtime={"max_rejected_root_fraction": 0.1})
    dataset, _, accounts = prepared_dataset(tmp_path / "dataset", config, monkeypatch)
    run = RunPaths(tmp_path / "run")
    saved_model(run.model, config, dataset)
    members = accounts[accounts.split == split]
    source = FakeSource(config, reject=frozenset({members.account_id.iloc[1]}))
    seen: list[tuple[int, int]] = []
    real = assemble.build_batch

    def record(store: Any, roots: list[ContextKey], **kwargs: Any) -> dict[str, torch.Tensor]:
        seen.extend((k.cutoff_seq, k.visibility_phase) for k in roots)
        return real(store, roots, **kwargs)

    monkeypatch.setattr(assemble, "build_batch", record)
    inputs = audit_inputs(run, dataset=dataset, hubs=hub_registry())
    truth = truth_of(members)
    # On a terminal the audit shows in place how much of its sample it has scored.
    monkeypatch.setattr(console, "is_terminal", lambda: True)
    result = audit(inputs, split, truth=truth, scope=split_scope(members, split), contexts=source)
    # The audit's line clears it.
    shown = [part.rstrip() for part in capsys.readouterr().out.split("\r")]
    assert shown[-3] == f"scoring the {split} audit sample {len(members)}/{len(members)}"
    assert shown[-1].startswith(f"Audited {split} at ")
    # The split's cutoff clock and phase.
    (date,) = DATES[split]
    assert set(seen) == {(CUTOFFS[date], SPLIT_PHASE[split])}
    assert result["split"] == split and result["date"] == date
    assert result["purpose"] == {"validation": "decisions", "test": "reporting"}[split]
    assert result["rejected_accounts"] == 1 and result["rejected_negatives"] == 1
    assert result["rejected"] == 1 and result["rejected_roots_by_status"] == {"missing_entity": 1}
    metrics = result["metrics"]
    assert metrics["sample_accounts"] == len(members) - 1
    assert metrics["evaluation_sample"] == (
        f"all_{split}_positives_plus_uniform_negatives_inverse_probability_weighted"
        "_minus_rejected_negatives"
    )
    # The report leads with the hidden mules. Nothing was revealed here, so they are every
    # mule, ranked alike.
    assert list(result)[5:9] == ["hidden_metrics", "hidden_intervals", "metrics", "intervals"]
    hidden = result["hidden_metrics"]
    assert hidden["evaluation_sample"] == metrics["evaluation_sample"].replace("all_", "hidden_")
    assert hidden["average_precision"] == metrics["average_precision"]
    assert result["hidden_intervals"] == result["intervals"]
    for name, interval in result["intervals"].items():
        assert interval is not None and interval[0] <= interval[1], name
    assert result["constants"] == {
        "audit_negatives": 2000,
        "sample_seed": config.dataset.split_seed,
        "review_budgets": [0.01, 0.05, 0.1],
        "interval": 0.9,
        "bootstrap_replicates": 1000,
        "bootstrap_seed": 0,
        "bootstrap": "positives_by_ring_negatives_within_class",
    }
    assert (result["revealed_positives"], result["hidden_positives"]) == (0, 6)
    assert read_json(run.audit_report(split)) == result
    scores = read_audit_scores(run.audit_scores(split))
    assert tuple(scores.columns) == AUDIT_COLUMNS
    assert scores.score.dtype == np.float64 and len(scores) == len(members) - 1
    expected = truth.set_index("account_id").loc[scores.account_id]
    assert scores.ring_id.tolist() == expected.ring_id.tolist()
    assert scores.label_source.tolist() == expected.label_source.tolist()
    assert not scores.revealed.any()
    assert run.audit_rejected(split).read_text().split() == [members.account_id.iloc[1]]
    # An audit the run already has is never overwritten.
    with pytest.raises(FileExistsError, match=f"{split}.json"):
        audit(inputs, split, truth=truth, scope=split_scope(members, split), contexts=source)
    with pytest.raises(ValueError, match="Audits cover"):
        audit(inputs, "train", truth=truth, scope=split_scope(members, split), contexts=source)


@pytest.mark.parametrize(("limit", "rejected_index"), [(1.0, 0), (0.0, 1)])
def test_the_audit_fails_on_censored_rejections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, limit: float, rejected_index: int
) -> None:
    config = unit_config(RUNTIME_CHANGES, runtime={"max_rejected_root_fraction": limit})
    dataset, _, accounts = prepared_dataset(tmp_path / "dataset", config, monkeypatch)
    run = RunPaths(tmp_path / "run")
    saved_model(run.model, config, dataset)
    members = accounts[accounts.split == "test"]
    # Index 0 is a test positive (always fatal); index 1 a negative (fatal at limit 0).
    source = FakeSource(config, reject=frozenset({members.account_id.iloc[rejected_index]}))
    match = r"\(1 test positive;" if rejected_index == 0 else r"\(0 test positives;"
    inputs = audit_inputs(run, dataset=dataset, hubs=hub_registry())
    with pytest.raises(ValueError, match=match):
        audit(
            inputs,
            "test",
            truth=truth_of(members),
            scope=split_scope(members, "test"),
            contexts=source,
        )
    assert not (run.root / "audit").exists()


def test_the_population_is_the_split_before_its_cutoff_with_what_the_graph_revealed() -> None:
    cutoff = timestamp("2025-01-01")

    def member(account: str, partition: int, first_seen: int, known: int) -> dict[str, Any]:
        return {
            "account_id": account,
            "partition": partition,
            "first_seen_ts_ms": first_seen,
            "observed_positive": known > 0,
            "known_from_ms": known,
        }

    rows = [
        member("A", 3, 1, 0),  # an account of the test split
        member("B", 3, 1, cutoff - 1),  # revealed just before the cutoff
        member("C", 3, 1, cutoff),  # revealed at the cutoff: not yet visible
        member("D", 3, cutoff, 0),  # first seen at the cutoff: not in the population
        member("E", 2, 1, 5),  # a validation account
    ]
    scope = TigerGraphScopeReader(FakeTigerGraph(population=rows))
    population = audit_population(scope, "unit_scope", "test", "2025-01-01")
    assert population.to_dict("list") == {
        "account_id": ["A", "B", "C"],
        "split": ["test"] * 3,
        "revealed": [False, True, False],
    }
    validation = audit_population(scope, "unit_scope", "validation", "2024-10-01")
    assert validation.to_dict("list") == {
        "account_id": ["E"],
        "split": ["validation"],
        "revealed": [True],
    }


def test_the_hidden_mules_are_ranked_with_the_revealed_ones_removed() -> None:
    # A revealed mule a on top, a non-mule c of weight 2, the hidden mule b, then the
    # non-mule d of weight 2.
    frame = pd.DataFrame(
        {
            "account_id": ["a", "b", "c", "d"],
            "is_mule": [1, 1, 0, 0],
            "inclusion_probability": [1.0, 1.0, 0.5, 0.5],
            "score": [0.9, 0.5, 0.7, 0.1],
            "revealed": [True, False, False, False],
            "ring_id": [-1, -1, -1, -1],
            "label_source": "phantomledger_role",
        }
    )
    results = audit_results(frame, 0.5, replicates=50)
    # Every mule: a, then c, then b: AP = 0.5 * 1 + 0.5 * 2 / 4.
    every = results["metrics"]
    assert every["average_precision"] == pytest.approx(0.75) and every["threshold"] == 0.5
    # As an investigator would, the hidden mules' ranking leaves the known a out: c, b, d.
    # b's precision is 1 / 3; it ranks above d and below c, so its ROC AUC is 0.5; the
    # population left is 5 accounts, a fifth of them the hidden mule.
    hidden = results["hidden_metrics"]
    assert hidden["average_precision"] == pytest.approx(1 / 3)
    assert hidden["roc_auc"] == pytest.approx(0.5)
    assert (hidden["sample_positives"], hidden["estimated_population"]) == (1, 5.0)
    assert hidden["weighted_prevalence"] == pytest.approx(0.2)
    # The top 10% is half an account, inside c: no hidden mule is found there.
    assert hidden["recall_at_10pct"] == 0.0 and "threshold" not in hidden
    assert set(results["hidden_intervals"]) == set(results["intervals"])
