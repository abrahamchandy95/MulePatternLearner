"""A suite's summary.csv and comparison.csv, and the paired comparison with the baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from mule_pattern_learner.artifacts import (
    AUDIT_COLUMNS,
    DELTA_METRIC,
    PAIRED_METRIC,
    read_comparison,
    read_json,
    read_run_provenance,
    read_summary,
    write_audit_scores,
    write_run_config,
)
from mule_pattern_learner.config import DEFAULT_CONFIG
from mule_pattern_learner.experiments.tables import (
    COMPARISON_COLUMNS,
    COMPLETE,
    FAILED,
    Delta,
    SuiteRun,
    paired_delta,
    paired_split,
    write_tables,
)
from mule_pattern_learner.experiments.variants import BASELINE, VARIANTS
from mule_pattern_learner.metrics import average_precision, paired_replicates, percentile_interval
from mule_pattern_learner.paths import RunPaths, SuitePaths
from mule_pattern_learner.testing.builders import write_suite_runs

# Six audited accounts: two mules of one ring, and four non-mules sampled at a half.
ACCOUNTS = ["a", "b", "c", "d", "e", "f"]
IS_MULE = [1, 1, 0, 0, 0, 0]
INCLUSION = [1.0, 1.0, 0.5, 0.5, 0.5, 0.5]
RINGS = [7, 7, -1, -1, -1, -1]


def audited_run(root: Path, name: str, seed: int, scores: list[float], drop: str = "") -> SuiteRun:
    """A run whose validation audit scored ACCOUNTS (all but drop) with these scores."""
    paths = RunPaths.of(name, seed, root)
    paths.audit_scores("validation").parent.mkdir(parents=True)
    frame = pd.DataFrame(
        {
            "account_id": ACCOUNTS,
            "is_mule": IS_MULE,
            "inclusion_probability": INCLUSION,
            "score": scores,
            "revealed": False,
            "ring_id": RINGS,
            "label_source": "role",
        }
    )
    write_audit_scores(paths.audit_scores("validation"), frame[frame.account_id != drop])
    return SuiteRun(VARIANTS[name], seed, paths, COMPLETE)


def test_the_paired_delta_of_a_small_case_worked_by_hand(tmp_path: Path) -> None:
    runs = [
        audited_run(tmp_path, "baseline", 1, [0.9, 0.8, 0.7, 0.1, 0.2, 0.3]),
        audited_run(tmp_path, "prior_weight", 1, [0.9, 0.2, 0.8, 0.1, 0.3, 0.4]),
        audited_run(tmp_path, "baseline", 2, [0.9, 0.8, 0.7, 0.1, 0.2, 0.3]),
        audited_run(tmp_path, "prior_weight", 2, [0.3, 0.9, 0.1, 0.2, 0.25, 0.05]),
    ]
    paired = paired_split(runs, "validation")
    assert paired is not None and paired.unpaired == 0
    # Seed 1 ranks a, then c, f and e (weight 2 each), then b: AP = 0.5 * 1 + 0.5 * 2 / 8.
    # Every other run ranks both mules first.
    assert paired.point.tolist() == pytest.approx([1.0, 0.625, 1.0, 1.0])
    delta = paired_delta(paired, "prior_weight")
    assert delta is not None and delta.seeds == pytest.approx({1: -0.375, 2: 0.0})
    assert delta.value == pytest.approx(-0.1875)
    # One seed without a difference: the delta is not consistent, whatever its interval.
    assert not delta.consistent
    # The interval is that of the mean difference over the replicates every run shares.
    y, weight = np.array(IS_MULE), 1 / np.array(INCLUSION)
    scores = [
        pd.read_parquet(run.paths.audit_scores("validation")).score.to_numpy() for run in runs
    ]
    replicates = paired_replicates(y, weight, scores, average_precision, np.array(RINGS))
    expected = replicates[:, [1, 3]].mean(axis=1) - replicates[:, [0, 2]].mean(axis=1)
    assert delta.interval == percentile_interval(expected)
    assert paired_delta(paired, "baseline") is None


def test_consistent_deltas_agree_in_sign_and_exclude_zero() -> None:
    assert Delta(-0.2, [-0.3, -0.1], {1: -0.1, 2: -0.3}).consistent
    assert Delta(0.2, [0.1, 0.3], {1: 0.1, 2: 0.3}).consistent
    assert not Delta(-0.2, [-0.3, 0.05], {1: -0.1, 2: -0.3}).consistent
    assert not Delta(-0.2, [-0.3, -0.1], {1: 0.1, 2: -0.5}).consistent
    assert not Delta(-0.2, None, {1: -0.1}).consistent


def test_accounts_some_audit_rejected_leave_the_pairing_and_other_samples_are_refused(
    tmp_path: Path,
) -> None:
    runs = [
        audited_run(tmp_path, "baseline", 1, [0.9, 0.8, 0.7, 0.1, 0.2, 0.3]),
        audited_run(tmp_path, "prior_weight", 1, [0.9, 0.2, 0.8, 0.1, 0.3, 0.4], drop="f"),
    ]
    paired = paired_split(runs, "validation")
    assert paired is not None and paired.unpaired == 1
    # Without f the variant ranks a, c, e (weight 2 each), then b: 0.5 + 0.5 * 2 / 6.
    assert paired.point.tolist() == pytest.approx([1.0, 0.5 + 0.5 * 2 / 6])
    other = audited_run(tmp_path, "no_attention", 1, [0.5] * 6)
    frame = pd.read_parquet(other.paths.audit_scores("validation"))
    frame.loc[frame.account_id == "c", "is_mule"] = 1
    write_audit_scores(other.paths.audit_scores("validation"), frame[list(AUDIT_COLUMNS)])
    with pytest.raises(ValueError, match="other accounts or truth"):
        paired_split([*runs, other], "validation")
    # Runs that did not complete are not paired.
    assert paired_split([SuiteRun(BASELINE, 1, RunPaths(tmp_path), FAILED)], "validation") is None


@pytest.fixture(scope="module")
def compared(tmp_path_factory: pytest.TempPathFactory) -> tuple[SuitePaths, list[SuiteRun]]:
    """A synthetic suite of three variants over two seeds, and one run that failed."""
    suite = SuitePaths.of("demo", tmp_path_factory.mktemp("results"))
    variants = [BASELINE, VARIANTS["no_attention"], VARIANTS["prior_weight"]]
    found = {"baseline": 0.62, "no_attention": 0.5, "prior_weight": 0.2}
    runs = write_suite_runs(suite, variants, (42, 43), found)
    # One run of another device, which comparison.csv flags.
    moved = runs[3].paths
    provenance = {**read_run_provenance(moved.config), "device": "cpu"}
    write_run_config(moved.config, VARIANTS["no_attention"].config(DEFAULT_CONFIG, 43), provenance)
    failed = SuiteRun(VARIANTS["no_slot_sum"], 42, suite.run("no_slot_sum", 42), FAILED, "boom")
    write_tables(suite, [*runs, failed])
    return suite, runs


def test_summary_csv_lists_every_run_split_and_metric(
    compared: tuple[SuitePaths, list[SuiteRun]],
) -> None:
    suite, runs = compared
    summary = read_summary(suite.summary)
    complete = summary[summary.status == COMPLETE]
    per_run = complete.groupby(["variant", "seed"]).size()
    # Three run values, the proxy AP, eight audit metrics of each split, the paired AP,
    # and the delta of every run but the baseline's.
    assert per_run.to_dict() == {
        (variant.name, seed): 21 if variant is BASELINE else 22
        for variant in (BASELINE, VARIANTS["no_attention"], VARIANTS["prior_weight"])
        for seed in (42, 43)
    }
    # A run that failed before writing anything keeps one row without a metric.
    failed = summary[summary.status == FAILED]
    assert failed[["variant", "seed", "split", "metric"]].values.tolist() == [
        ["no_slot_sum", 42, "", ""]
    ]
    assert failed.value.isna().all()
    # The recorded audit metrics, as each run's report holds them.
    run = runs[2]
    recorded = read_json(run.paths.audit_report("test"))["metrics"]["average_precision"]
    rows = complete[(complete.variant == run.variant.name) & (complete.seed == run.seed)]
    chosen = rows[(rows.split == "test") & (rows.metric == "average_precision")]
    assert chosen.value.tolist() == [recorded]
    assert set(rows.commit) == {"0" * 40}


def test_comparison_csv_compares_each_variant_with_the_baseline(
    compared: tuple[SuitePaths, list[SuiteRun]],
) -> None:
    suite, _ = compared
    table = read_comparison(suite.comparison)
    summary = read_summary(suite.summary)
    assert list(table.columns) == list(COMPARISON_COLUMNS)
    comparison: dict[str, dict[str, Any]] = {
        str(row["variant"]): {str(k): v for k, v in row.items()}
        for row in table.to_dict(orient="records")
    }
    assert list(comparison) == ["baseline", "no_attention", "prior_weight", "no_slot_sum"]
    for variant in ("no_attention", "prior_weight"):
        rows = summary[summary.variant == variant]
        validation = rows[(rows.split == "validation") & (rows.metric == "average_precision")]
        deltas = rows[rows.metric == DELTA_METRIC].value.to_numpy()
        row = comparison[variant]
        assert row["seeds"] == "42 43"
        assert row["validation_ap"] == pytest.approx(validation.value.mean())
        assert row["validation_ap_spread"] == pytest.approx(validation.value.std(ddof=1))
        # No audit rejected an account, so the paired AP is the recorded one.
        paired = rows[rows.metric == PAIRED_METRIC].value.to_numpy()
        assert paired == pytest.approx(validation.value.to_numpy())
        delta, low, high = (row[f"validation_ap_delta{end}"] for end in ("", "_low", "_high"))
        assert delta == pytest.approx(deltas.mean()) and low <= delta <= high
        assert row["validation_ap_low"] <= row["validation_ap"] <= row["validation_ap_high"]
        seeds = dict(enumerate(deltas.tolist()))
        assert row["consistent"] == Delta(delta, [low, high], seeds).consistent
    assert pd.isna(comparison["baseline"]["validation_ap_delta"])
    changes = "model.architecture = summary; model.slot_sum = False"
    assert comparison["no_attention"]["changes"] == changes
    assert comparison["no_attention"]["differs"] == "seed 43 device cpu"
    assert (
        comparison["baseline"]["differs"] == "" and comparison["baseline"]["unpaired_accounts"] == 0
    )
    # A variant without a complete run has no numbers.
    assert comparison["no_slot_sum"]["seeds"] == ""
    assert pd.isna(comparison["no_slot_sum"]["test_ap"])
