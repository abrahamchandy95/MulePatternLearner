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
    ENSEMBLE,
    ENSEMBLE_SEEDS,
    HIDDEN_METRIC,
    PAIRED_METRIC,
    SEED_MEAN,
    hidden_rows,
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
    EVERY,
    FAILED,
    HIDDEN,
    Delta,
    SuiteRun,
    paired_delta,
    paired_split,
    seed_ensembles,
    write_tables,
)
from mule_pattern_learner.experiments.variants import BASELINE, VARIANTS
from mule_pattern_learner.metrics import (
    average_precision,
    hidden_name,
    log_odds_mean,
    paired_replicates,
    percentile_interval,
    ranking_metrics,
    seed_draws,
    two_source_replicates,
)
from mule_pattern_learner.paths import RunPaths, SuitePaths
from mule_pattern_learner.testing.builders import write_suite_runs

# Six audited accounts: two mules of one ring, and four non-mules sampled at a half.
ACCOUNTS = ["a", "b", "c", "d", "e", "f"]
IS_MULE = [1, 1, 0, 0, 0, 0]
INCLUSION = [1.0, 1.0, 0.5, 0.5, 0.5, 0.5]
RINGS = [7, 7, -1, -1, -1, -1]


def audited_run(
    root: Path,
    name: str,
    seed: int,
    scores: list[float],
    drop: str = "",
    revealed: tuple[str, ...] = (),
) -> SuiteRun:
    """A run whose validation audit scored ACCOUNTS (all but drop) with these scores.

    The accounts ``revealed`` names are mules the graph revealed before the cutoff.
    """
    paths = RunPaths.of(name, seed, root)
    paths.audit_scores("validation").parent.mkdir(parents=True)
    frame = pd.DataFrame(
        {
            "account_id": ACCOUNTS,
            "is_mule": IS_MULE,
            "inclusion_probability": INCLUSION,
            "score": scores,
            "revealed": [account in revealed for account in ACCOUNTS],
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
    paired = paired_split(runs, "validation", EVERY)
    assert paired is not None and paired.unpaired == 0
    # Seed 1 ranks a, then c, f and e (weight 2 each), then b: AP = 0.5 * 1 + 0.5 * 2 / 8.
    # Every other run ranks both mules first.
    assert paired.point.tolist() == pytest.approx([1.0, 0.625, 1.0, 1.0])
    delta = paired_delta(paired, "prior_weight")
    assert delta is not None and delta.seeds == pytest.approx({1: -0.375, 2: 0.0})
    assert delta.value == pytest.approx(-0.1875)
    # Of the two seeds, only seed 1's delta has the sign of the mean.
    assert delta.agreeing == 1
    # The audit-only interval is that of the mean difference over the replicates every
    # run shares; the two-source one also resamples the seeds on each replicate.
    y, weight = np.array(IS_MULE), 1 / np.array(INCLUSION)
    scores = [
        pd.read_parquet(run.paths.audit_scores("validation")).score.to_numpy() for run in runs
    ]
    replicates = paired_replicates(y, weight, scores, average_precision, np.array(RINGS))
    differences = replicates[:, [1, 3]] - replicates[:, [0, 2]]
    assert delta.audit_interval == percentile_interval(differences.mean(axis=1))
    both = two_source_replicates(differences, seed_draws(2, replicates=len(differences)))
    assert delta.interval == percentile_interval(both)
    # Seed 2 differs nowhere, so a replicate that draws it twice has no difference, and
    # one that draws seed 1 twice has seed 1's: the seeds' spread widens the interval.
    assert delta.interval is not None and delta.audit_interval is not None
    low, high = delta.interval
    audit_low, audit_high = delta.audit_interval
    assert low <= audit_low and high >= audit_high and high == 0.0
    assert not delta.consistent
    assert paired_delta(paired, "baseline") is None


def test_the_two_source_replicates_of_a_small_case_worked_by_hand() -> None:
    # Three replicates of the accounts, two seeds: each row is one resample of the
    # accounts, each column a seed's difference from the baseline on it.
    differences = np.array([[1.0, 3.0], [2.0, 4.0], [0.0, 2.0]])
    # The first replicate draws seed 0 twice, the second seed 1 twice, the third both.
    draws = np.array([[0, 0], [1, 1], [0, 1]])
    assert two_source_replicates(differences, draws).tolist() == [1.0, 4.0, 1.0]
    # Without a resample of the seeds each replicate averages them all: the audit alone.
    assert two_source_replicates(differences, np.array([[0, 1]] * 3)).tolist() == [2, 3, 1]
    # The seeds' draws are their own: they do not repeat the accounts' resamples drawn
    # with the same seed, and they are the same on every call.
    first = seed_draws(10, replicates=1000)
    assert first.shape == (1000, 10) and first.min() == 0 and first.max() == 9
    assert (seed_draws(10, replicates=1000) == first).all()
    accounts = np.random.default_rng(0).integers(0, 10, size=(1000, 10))
    assert not (accounts == first).all()


def test_a_delta_is_consistent_when_its_interval_excludes_zero_and_every_seed_agrees() -> None:
    seeds = {1: -0.1, 2: -0.3}
    assert Delta(-0.2, [-0.3, -0.1], [-0.25, -0.15], seeds).consistent
    assert Delta(0.2, [0.1, 0.3], [0.15, 0.25], {1: 0.1, 2: 0.3}).consistent
    # The audit-only interval does not decide: the two-source one must exclude zero.
    assert not Delta(-0.2, [-0.3, 0.05], [-0.25, -0.15], seeds).consistent
    # A seed of the other sign is counted, and the delta is not consistent.
    mixed = Delta(-0.2, [-0.3, -0.1], [-0.25, -0.15], {1: 0.1, 2: -0.5, 3: -0.2})
    assert not mixed.consistent and mixed.agreeing == 2
    # Nor is one whose seed has no difference at all.
    assert not Delta(-0.2, [-0.3, -0.1], [-0.25, -0.15], {1: 0.0, 2: -0.4}).consistent
    assert not Delta(-0.2, None, None, {1: -0.1}).consistent


def test_accounts_some_audit_rejected_leave_the_pairing_and_other_samples_are_refused(
    tmp_path: Path,
) -> None:
    runs = [
        audited_run(tmp_path, "baseline", 1, [0.9, 0.8, 0.7, 0.1, 0.2, 0.3]),
        audited_run(tmp_path, "prior_weight", 1, [0.9, 0.2, 0.8, 0.1, 0.3, 0.4], drop="f"),
    ]
    paired = paired_split(runs, "validation", EVERY)
    assert paired is not None and paired.unpaired == 1
    # Without f the variant ranks a, c, e (weight 2 each), then b: 0.5 + 0.5 * 2 / 6.
    assert paired.point.tolist() == pytest.approx([1.0, 0.5 + 0.5 * 2 / 6])
    other = audited_run(tmp_path, "no_attention", 1, [0.5] * 6)
    frame = pd.read_parquet(other.paths.audit_scores("validation"))
    frame.loc[frame.account_id == "c", "is_mule"] = 1
    write_audit_scores(other.paths.audit_scores("validation"), frame[list(AUDIT_COLUMNS)])
    with pytest.raises(ValueError, match="other accounts or truth"):
        paired_split([*runs, other], "validation", EVERY)
    # Runs that did not complete are not paired.
    assert (
        paired_split([SuiteRun(BASELINE, 1, RunPaths(tmp_path), FAILED)], "validation", EVERY)
        is None
    )


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
    # Three run values, the proxy AP, eight audit metrics of each split of the hidden mules
    # and eight of every mule, the paired AP of each, and the deltas of every run but the
    # baseline's.
    assert per_run.to_dict() == {
        (variant.name, seed): 38 if variant is BASELINE else 40
        for variant in (BASELINE, VARIANTS["no_attention"], VARIANTS["prior_weight"])
        for seed in (42, 43)
    }
    # A run that failed before writing anything keeps one row without a metric.
    failed = summary[summary.status == FAILED]
    assert failed[["variant", "seed", "split", "metric"]].values.tolist() == [
        ["no_slot_sum", 42, "", ""]
    ]
    assert failed.value.isna().all()
    # The recorded audit metrics, as each run's report holds them, the hidden mules' named
    # with hidden_ first.
    run = runs[2]
    report = read_json(run.paths.audit_report("test"))
    rows = complete[(complete.variant == run.variant.name) & (complete.seed == run.seed)]
    for metric, recorded in (
        (HIDDEN_METRIC, report["hidden_metrics"]["average_precision"]),
        ("average_precision", report["metrics"]["average_precision"]),
    ):
        chosen = rows[(rows.split == "test") & (rows.metric == metric)]
        assert chosen.value.tolist() == [recorded]
    assert hidden_name("recall_at_1pct") in set(rows.metric)
    assert set(rows.commit) == {"0" * 40}


def test_comparison_csv_compares_each_variant_with_the_baseline(
    compared: tuple[SuitePaths, list[SuiteRun]],
) -> None:
    suite, _ = compared
    table = read_comparison(suite.comparison)
    summary = read_summary(suite.summary)
    assert list(table.columns) == list(COMPARISON_COLUMNS)
    # The seed means of every variant, then the ensembles of those with two seeds.
    assert table.estimate.tolist() == [SEED_MEAN] * 4 + [ENSEMBLE] * 3
    comparison: dict[str, dict[str, Any]] = {
        str(row["variant"]): {str(k): v for k, v in row.items()}
        for row in table[table.estimate == SEED_MEAN].to_dict(orient="records")
    }
    assert list(comparison) == ["baseline", "no_attention", "prior_weight", "no_slot_sum"]
    for variant in ("no_attention", "prior_weight"):
        rows = summary[(summary.variant == variant) & (summary.status == COMPLETE)]
        row = comparison[variant]
        assert row["seeds"] == "42 43"
        # The hidden mules' numbers and then every mule's, each from its own metrics.
        for prefix, mules in (("validation_hidden_ap", HIDDEN), ("validation_ap", EVERY)):
            name = hidden_name if mules == HIDDEN else str
            chosen = rows[(rows.split == "validation") & (rows.metric == name("average_precision"))]
            deltas = rows[rows.metric == name(DELTA_METRIC)].value.to_numpy()
            assert row[prefix] == pytest.approx(chosen.value.mean())
            assert row[f"{prefix}_spread"] == pytest.approx(chosen.value.std(ddof=1))
            # No audit rejected an account, so the paired AP is the recorded one.
            paired = rows[rows.metric == name(PAIRED_METRIC)].value.to_numpy()
            assert paired == pytest.approx(chosen.value.to_numpy())
            delta, low, high = (row[f"{prefix}_delta{end}"] for end in ("", "_low", "_high"))
            assert delta == pytest.approx(deltas.mean()) and low <= delta <= high
            assert row[f"{prefix}_low"] <= row[prefix] <= row[f"{prefix}_high"]
            seeds = dict(enumerate(deltas.tolist()))
            audit = [row[f"{prefix}_delta_audit_low"], row[f"{prefix}_delta_audit_high"]]
            assert audit[0] <= delta <= audit[1]
            found = Delta(delta, [low, high], audit, seeds)
            assert row[f"{prefix}_delta_seeds"] == 2
            assert row[f"{prefix}_delta_agreeing"] == found.agreeing
            # Decisions use the hidden mules: theirs is the delta that may be consistent.
            if mules == HIDDEN:
                assert row["consistent"] == found.consistent
        assert row["validation_hidden_ap"] < row["validation_ap"]
    assert pd.isna(comparison["baseline"]["validation_hidden_ap_delta"])
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


def test_each_variants_seed_ensemble_is_audited_as_a_run_is(
    compared: tuple[SuitePaths, list[SuiteRun]],
) -> None:
    suite, runs = compared
    table = read_comparison(suite.comparison)
    summary = read_summary(suite.summary)
    ensembles = {
        str(row["variant"]): {str(k): v for k, v in row.items()}
        for row in table[table.estimate == ENSEMBLE].to_dict(orient="records")
    }
    assert list(ensembles) == ["baseline", "no_attention", "prior_weight"]
    for split in ("validation", "test"):
        for mules in (HIDDEN, EVERY):
            paired = paired_split(runs, split, mules)
            assert paired is not None
            infix = "_hidden" if mules == HIDDEN else ""
            name = hidden_name if mules == HIDDEN else str
            for variant, row in ensembles.items():
                # The log-odds mean of the variant's two seeds on the shared accounts, audited.
                columns = list(paired.columns(variant).values())
                score = log_odds_mean(paired.scores[:, columns])
                expected = ranking_metrics(paired.y, score, paired.weight)
                ap = f"{split}{infix}_ap"
                assert row["seeds"] == "42 43" and pd.isna(row[f"{ap}_spread"])
                assert row[ap] == pytest.approx(expected["average_precision"])
                assert row[f"{split}{infix}_roc_auc"] == pytest.approx(expected["roc_auc"])
                assert row[f"{ap}_low"] <= row[ap] <= row[f"{ap}_high"]
                # summary.csv holds the same metrics, as rows of status ensemble without a
                # seed.
                rows = summary[(summary.status == ENSEMBLE) & (summary.variant == variant)]
                assert rows.seed.isna().all() and set(rows.commit) == {""}
                metric = name("average_precision")
                chosen = rows[(rows.split == split) & (rows.metric == metric)]
                assert chosen.value.tolist() == pytest.approx([expected["average_precision"]])
                seeds = rows[rows.metric == ENSEMBLE_SEEDS].value.tolist()
                assert seeds == [2.0]
    # Ensembles have no delta, no consistency and no run values of their own.
    for row in ensembles.values():
        assert pd.isna(row["validation_hidden_ap_delta"]) and pd.isna(row["consistent"])
        assert pd.isna(row["validation_ap_delta"]) and pd.isna(row["best_epoch"])


def test_a_seed_ensemble_of_a_small_case_worked_by_hand(tmp_path: Path) -> None:
    runs = [
        audited_run(tmp_path, "baseline", 1, [0.9, 0.8, 0.7, 0.1, 0.2, 0.3]),
        audited_run(tmp_path, "prior_weight", 1, [0.9, 0.2, 0.8, 0.1, 0.3, 0.4]),
        audited_run(tmp_path, "baseline", 2, [0.9, 0.8, 0.7, 0.1, 0.2, 0.3]),
        audited_run(tmp_path, "prior_weight", 2, [0.3, 0.9, 0.1, 0.2, 0.25, 0.05]),
        audited_run(tmp_path, "no_attention", 1, [0.5] * 6),
    ]
    paired = paired_split(runs, "validation", EVERY)
    assert paired is not None
    ensembles = seed_ensembles(paired)
    # A variant of one seed has no ensemble.
    assert set(ensembles) == {"baseline", "prior_weight"}
    # The baseline's two seeds agree, so its ensemble is either of them: AP 1.
    assert ensembles["baseline"].metrics["average_precision"] == pytest.approx(1.0)
    # prior_weight's ensemble takes the geometric mean of each account's odds over its
    # seeds: a 9 and 3/7 make 1.96, b 1/4 and 9 make 1.5, c 4 and 1/9 make 0.67, then e
    # 0.38, f 0.19 and d 0.17. Both mules rank first, so its AP is 1, where its seeds'
    # are 0.625 and 1.
    odds = np.sqrt([9 * 3 / 7, 0.25 * 9, 4 / 9, 1 / 9 * 0.25, 3 / 7 / 3, 2 / 3 / 19])
    assert log_odds_mean(paired.scores[:, [1, 3]]) == pytest.approx(odds / (1 + odds))
    ensemble = ensembles["prior_weight"]
    assert ensemble.seeds == (1, 2)
    assert ensemble.metrics["average_precision"] == pytest.approx(1.0)
    # Its interval is that of its AP over the replicates every run of the split shares.
    y, weight = np.array(IS_MULE), 1 / np.array(INCLUSION)
    replicates = paired_replicates(
        y, weight, [odds / (1 + odds)], average_precision, np.array(RINGS)
    )
    assert ensemble.interval == pytest.approx(percentile_interval(replicates[:, 0]))


def test_the_log_odds_mean_lets_a_confident_seed_weigh_more_than_a_rank_mean_would() -> None:
    # Two seeds of two accounts: the first seed is sure of a and lukewarm on b, the second
    # mildly prefers b. The ranks tie, a rank mean cannot separate them, but on the
    # log-odds scale the confident seed carries a above b.
    scores = np.array([[0.999999, 0.4], [0.6, 0.6]])
    combined = log_odds_mean(scores)
    assert combined[0] > combined[1]
    assert combined[1] == pytest.approx(0.6)
    # Two scores of 0.9 and 0.5 have odds 9 and 1, whose geometric mean 3 is a score of 0.75.
    assert log_odds_mean(np.array([[0.9, 0.5]])) == pytest.approx([0.75])
    # Scores of exactly 0 or 1 are clipped, so the mean stays finite and inside (0, 1).
    clipped = log_odds_mean(np.array([[1.0, 1.0], [0.0, 0.0], [1.0, 0.0]]))
    assert np.isfinite(clipped).all() and 0 < clipped[1] < clipped[2] < clipped[0] < 1
    assert clipped[2] == pytest.approx(0.5)


def test_the_hidden_mules_are_paired_with_the_revealed_ones_left_out(tmp_path: Path) -> None:
    # Mule a was revealed. The baseline ranks the hidden mule b above every non-mule:
    # AP 1. prior_weight ranks c, f and e (weight 2 each) above b: AP 1 / (1 + 6).
    revealed = ("a",)
    baseline = audited_run(tmp_path, "baseline", 1, [0.9, 0.8, 0.7, 0.1, 0.2, 0.3], "", revealed)
    other = [0.9, 0.2, 0.8, 0.1, 0.3, 0.4]
    variant = audited_run(tmp_path, "prior_weight", 1, other, "", revealed)
    paired = paired_split([baseline, variant], "validation", HIDDEN)
    assert paired is not None and paired.mules == HIDDEN
    assert paired.point.tolist() == pytest.approx([1.0, 1 / 7])
    # The accounts left are the audit sample's hidden view: every account but a.
    frame = pd.read_parquet(variant.paths.audit_scores("validation"))
    assert paired.y.tolist() == hidden_rows(frame).is_mule.tolist()
    delta = paired_delta(paired, "prior_weight")
    assert delta is not None and delta.value == pytest.approx(1 / 7 - 1)
    # Every mule, a included, ranks it first in both runs.
    every = paired_split([baseline, variant], "validation", EVERY)
    assert every is not None and len(every.y) == len(paired.y) + 1
    # Without a hidden mule there is no AP of them.
    both = audited_run(tmp_path, "no_attention", 1, other, "", ("a", "b"))
    alone = paired_split([both], "validation", HIDDEN)
    assert alone is not None and np.isnan(alone.point).all()
    # Audits that disagree on what was revealed belong to other samples.
    with pytest.raises(ValueError, match="other accounts or truth"):
        paired_split([baseline, both], "validation", HIDDEN)
