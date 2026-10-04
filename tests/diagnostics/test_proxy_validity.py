"""Proxy validity: the run's ranking against the truth, the hidden mules from its audit."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mule_pattern_learner.artifacts import (
    AUDIT_COLUMNS,
    write_audit_scores,
    write_json,
    write_predictions,
)
from mule_pattern_learner.diagnostics.proxy_validity import COLUMNS, SUBSETS, proxy_validity
from mule_pattern_learner.paths import RunPaths

# One split's proxy predictions: the revealed mule R, three non-mules (N1 to N3) and an
# account whose truth is unknown (U). Like most proxy samples of a rare class, they hold
# no hidden mule.
ROWS = [("R", 1, 0.9), ("N1", 0, 0.8), ("U", 0, 0.5), ("N2", 0, 0.2), ("N3", 0, 0.1)]
TRUTH = pd.DataFrame(
    {
        "account_id": ["R", "H1", "H2", "N1", "N2", "N3", "U"],
        "is_mule": [1, 1, 1, 0, 0, 0, -1],
        "ring_id": [0, 0, -1, -1, -1, -1, -1],
        "label_source": "phantomledger_role",
    }
)
# The split's audit sample: every mule, R revealed and the hidden H1 and H2, and the
# non-mules, each standing for two accounts.
AUDIT = pd.DataFrame(
    {
        "account_id": ["R", "N1", "H1", "H2", "N2", "N3"],
        "is_mule": [1, 0, 1, 1, 0, 0],
        "inclusion_probability": [1.0, 0.5, 1.0, 1.0, 0.5, 0.5],
        "score": [0.9, 0.8, 0.7, 0.3, 0.2, 0.1],
        "revealed": [True, False, False, False, False, False],
        "ring_id": [0, -1, 0, -1, -1, -1],
        "label_source": "phantomledger_role",
    }
)[list(AUDIT_COLUMNS)]


def predictions(date: str) -> pd.DataFrame:
    frame = pd.DataFrame(ROWS, columns=["account_id", "observed_label", "score"])
    frame.insert(1, "group_id", "G" + frame.account_id)
    frame.insert(2, "date", date)
    return frame


def run_of(root: Path, audited: tuple[str, ...] = ("validation", "test")) -> RunPaths:
    """A complete run's metrics.json and proxy predictions, and the audits named."""
    run = RunPaths(root / "run")
    run.root.mkdir()
    write_json(run.metrics, {"validation_proxy": {"threshold": 0.75}})
    for split, date in (("validation", "2024-10-01"), ("test", "2025-01-01")):
        write_predictions(run.predictions(split), predictions(date))
        if split in audited:
            run.audit_report(split).parent.mkdir(exist_ok=True)
            write_audit_scores(run.audit_scores(split), AUDIT)
            write_json(run.audit_report(split), {"split": split})
    return run


def test_the_hidden_and_revealed_mules_come_from_the_audit_and_all_from_the_proxy(
    tmp_path: Path,
) -> None:
    table = proxy_validity(run_of(tmp_path), TRUTH)
    assert tuple(table.columns) == COLUMNS
    assert set(table.split) == {"validation", "test"}
    assert table.subset.unique().tolist() == list(SUBSETS) == ["hidden", "revealed", "all"]
    values = table[table.split == "test"].set_index(["subset", "metric"]).value
    # The proxy's sample holds no hidden mule, the audit sample every one. Without R it
    # ranks N1 (two accounts), H1, H2, N2 and N3 (two each): H1's precision is 1/3 and
    # H2's 2/4, so AP = 5/12; each hidden mule ranks above 4 of the 6 non-mule accounts.
    assert values["hidden", "n"] == 5 and values["hidden", "positives"] == 2
    assert values["hidden", "prevalence"] == pytest.approx(2 / 8)
    assert values["hidden", "average_precision"] == pytest.approx(5 / 12)
    assert values["hidden", "roc_auc"] == pytest.approx(2 / 3)
    # The threshold 0.75 flags N1 alone among them.
    assert values["hidden", "recall"] == 0 and values["hidden", "precision"] == 0
    # Revealed: R above every non-mule, the hidden mules left out.
    assert values["revealed", "n"] == 4 and values["revealed", "positives"] == 1
    assert values["revealed", "average_precision"] == 1 and values["revealed", "roc_auc"] == 1
    # All: the four predicted accounts of known truth, unweighted, R first.
    assert values["all", "n"] == 4 and values["all", "positives"] == 1
    assert values["all", "unknown_truth"] == 1
    assert values["all", "average_precision"] == 1 and values["all", "recall"] == 1
    # The splits are scored alike.
    validation = table[table.split == "validation"].value.to_numpy()
    assert (validation == table[table.split == "test"].value.to_numpy()).all()


def test_only_the_audited_splits_are_scored(tmp_path: Path) -> None:
    table = proxy_validity(run_of(tmp_path, audited=("validation",)), TRUTH)
    assert set(table.split) == {"validation"}


def test_an_observed_positive_that_is_no_mule_breaks_the_label_contract(tmp_path: Path) -> None:
    wrong = TRUTH.assign(is_mule=TRUTH.is_mule.where(TRUTH.account_id != "R", 0))
    with pytest.raises(ValueError, match="label contract"):
        proxy_validity(run_of(tmp_path), wrong)
