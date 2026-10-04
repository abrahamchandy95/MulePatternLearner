"""A suite's comparison tables: summary.csv, every run's numbers, and comparison.csv.

summary.csv is long: one row per run, split and metric (SUMMARY_COLUMNS), with the run's
status and commit on every row. A run's audit metrics come from its audit reports, its
proxy AP and totals from metrics.json; values with no split belong to the run as a
whole. A run that left no numbers keeps one row without a metric, so every run of the
suite is listed.

comparison.csv holds one row per variant (COMPARISON_COLUMNS): the seed means of the
audit metrics, their spread over seeds, and the paired comparison with the baseline.
Every variant is audited on the same accounts, because the audit sample depends only on
the scope, the truth and dataset.split_seed. Each bootstrap replicate therefore draws one
resample of those accounts and their rings and applies it to every run
(metrics.paired_replicates), and the seed-mean AP of a variant gets its 90% interval
from those replicates: the audit sample's uncertainty for these seeds. The variant's
difference from the baseline's (over the seeds both completed, paired by seed) gets two:
the audit-only interval, from the same replicates, and the two-source interval, which
on each replicate also resamples the seeds (metrics.two_source_replicates), so it covers
the spread between seeds as well. Decisions use the two-source interval: a delta is
consistent when it excludes zero, and comparison.csv counts the seeds whose own delta
has the mean's sign. Accounts some run's audit rejected are left out of the pairing,
and comparison.csv counts them.

A variant's seed ensemble combines its complete seeds (two or more) into one ranking of
those shared accounts, their scores averaged on the log-odds scale
(metrics.log_odds_mean, whose docstring gives the trade-off against averaging ranks).
Each ensemble is audited as a run is, on validation and test: the ranking metrics of
the audit, and the AP's 90% interval over the same replicates. summary.csv gives its
metrics in rows of status "ensemble" without a seed, and comparison.csv a row of its own
after the seed means (estimate "ensemble" against "seed_mean").
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from ..artifacts import (
    DELTA_METRIC,
    ENSEMBLE,
    ENSEMBLE_SEEDS,
    PAIRED_METRIC,
    PROXY_METRIC,
    SEED_MEAN,
    SUMMARY_COLUMNS,
    read_audit_scores,
    read_json,
    read_run_provenance,
    write_table,
)
from ..config import DEFAULT_CONFIG, RunConfig
from ..contract.graph_schema import HELD_OUT_SPLITS
from ..metrics import (
    REVIEW_BUDGETS,
    average_precision,
    budget_name,
    log_odds_mean,
    paired_replicates,
    percentile_interval,
    ranking_metrics,
    two_source_replicates,
)
from ..paths import BASELINE_VARIANT, RunPaths, SuitePaths
from .variants import Variant

# What became of a run of the suite: trained and audited, failed with an error of its
# own, or not trained or audited because an outage stopped the suite.
COMPLETE, FAILED, STOPPED = "complete", "failed", "stopped"
# The audit metrics of each split, as the audit reports name them.
BUDGET_METRICS = tuple(
    f"{kind}_at_{budget_name(fraction)}"
    for fraction in REVIEW_BUDGETS
    for kind in ("recall", "precision")
)
AUDIT_METRICS = ("average_precision", "roc_auc", *BUDGET_METRICS)
# The run's own values, with no split.
RUN_METRICS = ("best_epoch", "parameter_count", "training_hours")
# What comparison.csv flags when a variant's runs differ from the suite's usual value.
PROVENANCE = ("git_commit", "git_dirty", "device", "sampler_backend")


def _split_columns(split: str) -> list[str]:
    return [
        f"{split}_ap",
        f"{split}_ap_spread",
        f"{split}_ap_low",
        f"{split}_ap_high",
    ]


COMPARISON_COLUMNS = (
    "variant",
    "estimate",
    "question",
    "changes",
    "seeds",
    *(column for split in HELD_OUT_SPLITS for column in _split_columns(split)),
    "validation_ap_delta",
    "validation_ap_delta_low",
    "validation_ap_delta_high",
    "validation_ap_delta_audit_low",
    "validation_ap_delta_audit_high",
    "validation_ap_delta_seeds",
    "validation_ap_delta_agreeing",
    "consistent",
    *(f"{split}_{metric}" for split in HELD_OUT_SPLITS for metric in AUDIT_METRICS[1:]),
    *RUN_METRICS,
    "unpaired_accounts",
    "differs",
)


@dataclass(frozen=True)
class SuiteRun:
    """One run of a suite: its variant, seed and directory, and what became of it."""

    variant: Variant
    seed: int
    paths: RunPaths
    status: str
    error: str | None = None


def audited(run: RunPaths) -> bool:
    """Whether a run is complete and audited on every held-out split."""
    reports = (run.audit_report(split) for split in HELD_OUT_SPLITS)
    return run.metrics.exists() and all(path.exists() for path in reports)


def provenance(run: RunPaths) -> dict[str, Any]:
    """The provenance config.json records, or nothing for a run that never started."""
    return read_run_provenance(run.config) if run.config.exists() else {}


def run_values(run: RunPaths) -> list[tuple[str, str, float]]:
    """(split, metric, value) of every number a run's files hold for summary.csv."""
    values: list[tuple[str, str, float]] = []
    if run.metrics.exists():
        metrics = read_json(run.metrics)
        values += [
            ("", "best_epoch", float(metrics["best_epoch"])),
            ("", "parameter_count", float(metrics["parameter_count"])),
            ("", "training_hours", float(metrics["elapsed_seconds"]) / 3600),
        ]
        proxy = metrics["validation_proxy"].get("average_precision")
        if proxy is not None:
            values.append(("validation", PROXY_METRIC, float(proxy)))
    for split in HELD_OUT_SPLITS:
        if run.audit_report(split).exists():
            recorded = read_json(run.audit_report(split))["metrics"]
            values += [
                (split, name, float(recorded[name]))
                for name in AUDIT_METRICS
                if recorded.get(name) is not None
            ]
    return values


@dataclass(frozen=True)
class PairedSplit:
    """The runs' AP on the accounts every audit of a split scored, and on each replicate.

    ``point`` holds each run's AP on those accounts and ``replicates`` its AP on each
    shared bootstrap replicate (replicates by runs, in the order of ``runs``).
    ``unpaired`` counts the accounts some audit rejected, which are left out. ``y``,
    ``weight`` and ``rings`` are the shared accounts' truth, inverse inclusion
    probability and ring, and ``scores`` each run's scores of them (accounts by runs).
    """

    runs: tuple[SuiteRun, ...]
    point: NDArray[np.float64]
    replicates: NDArray[np.float64]
    unpaired: int
    y: NDArray[np.int64]
    weight: NDArray[np.float64]
    rings: NDArray[np.int64]
    scores: NDArray[np.float64]

    def columns(self, variant: str) -> dict[int, int]:
        """The column of each of a variant's runs, by seed."""
        return {run.seed: i for i, run in enumerate(self.runs) if run.variant.name == variant}

    def mean_interval(self, variant: str) -> list[float] | None:
        """The interval of a variant's seed-mean AP over the shared replicates."""
        columns = list(self.columns(variant).values())
        if not columns:
            return None
        return percentile_interval(self.replicates[:, columns].mean(axis=1))


def paired_split(runs: Sequence[SuiteRun], split: str) -> PairedSplit | None:
    """The paired AP of the complete runs on a split's shared audit accounts.

    Every audit scored the same sample, less the accounts TigerGraph rejected in it, so
    the accounts every run scored carry the same truth, inclusion probability and ring in
    each audit; an audit that disagrees belongs to another sample and is refused. None
    without complete runs.
    """
    complete = tuple(run for run in runs if run.status == COMPLETE)
    if not complete:
        return None
    frames = [
        read_audit_scores(run.paths.audit_scores(split)).set_index("account_id") for run in complete
    ]
    every = set[str]().union(*(set(frame.index) for frame in frames))
    shared = sorted(set[str](frames[0].index).intersection(*(frame.index for frame in frames)))
    aligned = [frame.loc[shared] for frame in frames]
    truth = aligned[0][["is_mule", "inclusion_probability", "ring_id"]]
    for run, frame in zip(complete, aligned, strict=True):
        if not frame[truth.columns].equals(truth):
            raise ValueError(
                f"The {split} audit of {run.paths.root} scored other accounts or truth than "
                f"the suite's other audits; audits of one sample are needed to pair them"
            )
    y = truth.is_mule.to_numpy(np.int64)
    weight = 1 / truth.inclusion_probability.to_numpy(np.float64)
    rings = truth.ring_id.to_numpy(np.int64)
    scores = [frame.score.to_numpy(np.float64) for frame in aligned]
    point = np.array(
        [np.nan if (ap := average_precision(y, s, weight)) is None else ap for s in scores]
    )
    replicates = paired_replicates(y, weight, scores, average_precision, rings)
    unpaired = len(every) - len(shared)
    matrix = np.column_stack(scores) if scores else np.zeros((0, 0))
    return PairedSplit(complete, point, replicates, unpaired, y, weight, rings, matrix)


@dataclass(frozen=True)
class Ensemble:
    """A variant's seed ensemble on a split's shared accounts, audited as a run is.

    ``seeds`` are the seeds it combines, ``metrics`` the audit's ranking metrics
    (metrics.ranking_metrics) and ``interval`` the 90% interval of its AP over the
    split's shared replicates.
    """

    seeds: tuple[int, ...]
    metrics: dict[str, float | None]
    interval: list[float] | None


def seed_ensembles(paired: PairedSplit) -> dict[str, Ensemble]:
    """The seed ensemble of every variant with two or more complete runs on a split.

    Each averages its seeds' scores of the shared accounts on the log-odds scale
    (metrics.log_odds_mean). Its AP interval comes from the bootstrap replicates every
    run of the split shares (one resample of the accounts and rings per replicate,
    metrics.paired_replicates), the replicates of every audit's own interval.
    """
    combined: dict[str, tuple[tuple[int, ...], NDArray[np.float64]]] = {}
    for variant in dict.fromkeys(run.variant.name for run in paired.runs):
        columns = paired.columns(variant)
        if len(columns) > 1:
            score = log_odds_mean(paired.scores[:, list(columns.values())])
            combined[variant] = (tuple(columns), score)
    if not combined:
        return {}
    scores = [score for _, score in combined.values()]
    replicates = paired_replicates(paired.y, paired.weight, scores, average_precision, paired.rings)
    return {
        variant: Ensemble(
            seeds,
            ranking_metrics(paired.y, score, paired.weight),
            percentile_interval(replicates[:, column]),
        )
        for column, (variant, (seeds, score)) in enumerate(combined.items())
    }


@dataclass(frozen=True)
class Delta:
    """A variant's paired difference in validation AP from the baseline's.

    ``interval`` covers both sources of uncertainty, the seeds and the audit sample
    (metrics.two_source_replicates); ``audit_interval`` the audit sample's alone, for
    these seeds. ``seeds`` holds each seed's difference.
    """

    value: float
    interval: list[float] | None
    audit_interval: list[float] | None
    seeds: dict[int, float]

    @property
    def consistent(self) -> bool:
        """Whether the two-source interval excludes zero."""
        if self.interval is None:
            return False
        low, high = self.interval
        return low > 0 or high < 0

    @property
    def agreeing(self) -> int:
        """The seeds whose difference has the sign of the mean difference."""
        return sum(delta * self.value > 0 for delta in self.seeds.values())


def paired_delta(paired: PairedSplit, variant: str) -> Delta | None:
    """The variant's seed-mean AP minus the baseline's, over the seeds both completed.

    Each seed's difference pairs the variant's run with the baseline's run of that seed,
    on the accounts every audit scored. The audit-only interval averages the seeds'
    differences on each shared replicate of those accounts; the two-source interval also
    resamples the seeds on each replicate (metrics.two_source_replicates), so it widens
    with the spread between seeds. None for the baseline itself and for a variant with no
    seed the baseline completed.
    """
    mine, baseline = paired.columns(variant), paired.columns(BASELINE_VARIANT)
    seeds = sorted(mine.keys() & baseline.keys())
    if variant == BASELINE_VARIANT or not seeds:
        return None
    ours, theirs = [mine[s] for s in seeds], [baseline[s] for s in seeds]
    by_seed = {s: float(paired.point[mine[s]] - paired.point[baseline[s]]) for s in seeds}
    differences = paired.replicates[:, ours] - paired.replicates[:, theirs]
    value = float(paired.point[ours].mean() - paired.point[theirs].mean())
    return Delta(
        value,
        percentile_interval(two_source_replicates(differences)),
        percentile_interval(differences.mean(axis=1)),
        by_seed,
    )


def summary_rows(
    runs: Sequence[SuiteRun],
    validation: PairedSplit | None,
    ensembles: Mapping[str, Mapping[str, Ensemble]],
) -> list[dict[str, Any]]:
    """summary.csv's rows: each run's numbers and paired validation AP and delta, then ensembles.

    ``ensembles`` holds each split's seed ensembles by variant (seed_ensembles).
    """
    deltas: dict[tuple[str, int], float] = {}
    paired_ap: dict[tuple[str, int], float] = {}
    if validation is not None:
        paired_ap = {
            (run.variant.name, run.seed): float(value)
            for run, value in zip(validation.runs, validation.point, strict=True)
        }
        for variant in dict.fromkeys(run.variant.name for run in validation.runs):
            delta = paired_delta(validation, variant)
            if delta is not None:
                deltas |= {(variant, seed): value for seed, value in delta.seeds.items()}
    rows: list[dict[str, Any]] = []
    for run in runs:
        key = (run.variant.name, run.seed)
        values = run_values(run.paths)
        if key in paired_ap:
            values.append(("validation", PAIRED_METRIC, paired_ap[key]))
        if key in deltas:
            values.append(("validation", DELTA_METRIC, deltas[key]))
        commit = provenance(run.paths).get("git_commit") or ""
        fixed = {"variant": run.variant.name, "seed": run.seed, "status": run.status}
        if not values:
            rows.append({**fixed, "split": "", "metric": "", "value": np.nan, "commit": commit})
        for split, metric, value in values:
            rows.append(
                {**fixed, "split": split, "metric": metric, "value": value, "commit": commit}
            )
    for variant in dict.fromkeys(run.variant.name for run in runs):
        found = {split: by[variant] for split, by in ensembles.items() if variant in by}
        if not found:
            continue
        fixed = {"variant": variant, "seed": None, "status": ENSEMBLE, "commit": ""}
        seeds = max(len(ensemble.seeds) for ensemble in found.values())
        values = [("", ENSEMBLE_SEEDS, float(seeds))]
        values += [
            (split, name, value)
            for split, ensemble in found.items()
            for name, value in ensemble.metrics.items()
            if value is not None
        ]
        for split, metric, value in values:
            rows.append({**fixed, "split": split, "metric": metric, "value": value})
    return [{name: row[name] for name in SUMMARY_COLUMNS} for row in rows]


def _usual(values: Sequence[Any]) -> Any:
    """The most common value, the first of those tied."""
    return Counter(values).most_common(1)[0][0] if values else None


def differences(runs: Sequence[SuiteRun]) -> dict[str, str]:
    """For each variant, which of its complete runs differ from the suite's usual provenance.

    The usual value of each PROVENANCE key is the most common among the complete runs;
    a variant's entry names the seed and the key of every run that differs.
    """
    complete = [(run, provenance(run.paths)) for run in runs if run.status == COMPLETE]
    usual = {key: _usual([recorded.get(key) for _, recorded in complete]) for key in PROVENANCE}
    notes: dict[str, list[str]] = {}
    for run, recorded in complete:
        for key in PROVENANCE:
            value = recorded.get(key)
            if value != usual[key]:
                shown = str(value)[:12] if key == "git_commit" else value
                notes.setdefault(run.variant.name, []).append(f"seed {run.seed} {key} {shown}")
    return {variant: "; ".join(items) for variant, items in notes.items()}


def seed_means(
    values: Sequence[Mapping[tuple[str, str], float]], split: str, metric: str
) -> tuple[float, float]:
    """The mean over seeds of a metric the runs recorded, and its standard deviation.

    ``values`` are the (split, metric) values of each of a variant's complete runs. The
    standard deviation needs two seeds; a value missing everywhere is NaN.
    """
    found = [run[split, metric] for run in values if (split, metric) in run]
    mean = float(np.mean(found)) if found else np.nan
    return mean, float(np.std(found, ddof=1)) if len(found) > 1 else np.nan


def interval_pair(interval: list[float] | None) -> tuple[float, float]:
    return (interval[0], interval[1]) if interval is not None else (np.nan, np.nan)


def comparison_rows(
    runs: Sequence[SuiteRun],
    paired: Mapping[str, PairedSplit | None],
    ensembles: Mapping[str, Mapping[str, Ensemble]],
    base: RunConfig,
) -> list[dict[str, Any]]:
    """comparison.csv's rows: one per variant, in the suite's order, then one per ensemble.

    Point values are the seed means of what each complete run's audit recorded; the
    intervals and the delta come from the paired AP of the shared accounts. An ensemble's
    row holds its own audit metrics and AP intervals (``ensembles``, by split and variant).
    """
    complete = [run for run in runs if run.status == COMPLETE]
    recorded = {
        id(run): {(split, metric): value for split, metric, value in run_values(run.paths)}
        for run in complete
    }
    flags = differences(runs)
    validation = paired["validation"]
    rows = []
    for variant in dict.fromkeys(run.variant for run in runs):
        mine = [run for run in complete if run.variant.name == variant.name]
        values = [recorded[id(run)] for run in mine]
        row: dict[str, Any] = {
            "variant": variant.name,
            "estimate": SEED_MEAN,
            "question": variant.question,
            "changes": variant.change_text(base),
            "seeds": " ".join(str(run.seed) for run in mine),
        }
        for split in HELD_OUT_SPLITS:
            pairing = paired[split]
            interval = pairing.mean_interval(variant.name) if pairing is not None else None
            cells = (*seed_means(values, split, "average_precision"), *interval_pair(interval))
            row |= dict(zip(_split_columns(split), cells, strict=True))
        delta = paired_delta(validation, variant.name) if validation is not None else None
        if delta is not None:
            low, high = interval_pair(delta.interval)
            audit_low, audit_high = interval_pair(delta.audit_interval)
            row |= {
                "validation_ap_delta": delta.value,
                "validation_ap_delta_low": low,
                "validation_ap_delta_high": high,
                "validation_ap_delta_audit_low": audit_low,
                "validation_ap_delta_audit_high": audit_high,
                "validation_ap_delta_seeds": len(delta.seeds),
                "validation_ap_delta_agreeing": delta.agreeing,
                "consistent": delta.consistent,
            }
        for split in HELD_OUT_SPLITS:
            for metric in AUDIT_METRICS[1:]:
                row[f"{split}_{metric}"] = seed_means(values, split, metric)[0]
        row |= {metric: seed_means(values, "", metric)[0] for metric in RUN_METRICS}
        row["unpaired_accounts"] = validation.unpaired if validation is not None else np.nan
        row["differs"] = flags.get(variant.name, "")
        rows.append({name: row.get(name, np.nan) for name in COMPARISON_COLUMNS})
    for variant in dict.fromkeys(run.variant for run in runs):
        found = {split: by[variant.name] for split, by in ensembles.items() if variant.name in by}
        if not found:
            continue
        seeds = max((ensemble.seeds for ensemble in found.values()), key=len)
        row = {
            "variant": variant.name,
            "estimate": ENSEMBLE,
            "question": variant.question,
            "changes": variant.change_text(base),
            "seeds": " ".join(map(str, seeds)),
        }
        for split, ensemble in found.items():
            low, high = interval_pair(ensemble.interval)
            row |= {f"{split}_ap": ensemble.metrics["average_precision"]}
            row |= {f"{split}_ap_low": low, f"{split}_ap_high": high}
            for metric in AUDIT_METRICS[1:]:
                row[f"{split}_{metric}"] = ensemble.metrics.get(metric)
        row["unpaired_accounts"] = validation.unpaired if validation is not None else np.nan
        row["differs"] = flags.get(variant.name, "")
        rows.append({name: _cell(row.get(name)) for name in COMPARISON_COLUMNS})
    return rows


def _cell(value: Any) -> Any:
    """A table cell: a value, or NaN for one that is missing."""
    return np.nan if value is None else value


def write_tables(
    suite: SuitePaths, runs: Sequence[SuiteRun], base: RunConfig = DEFAULT_CONFIG
) -> dict[str, Any]:
    """Write the suite's summary.csv and comparison.csv from its runs' files.

    ``runs`` are every run of the suite, in its order, each with its status; only the
    complete ones enter the comparison. ``base`` is the run the variants change, which
    comparison.csv's changes are stated against. Returns both paths and the accounts
    left out of the validation pairing.
    """
    suite.root.mkdir(parents=True, exist_ok=True)
    paired = {split: paired_split(runs, split) for split in HELD_OUT_SPLITS}
    ensembles = {
        split: seed_ensembles(pairing) for split, pairing in paired.items() if pairing is not None
    }
    summary = summary_rows(runs, paired["validation"], ensembles)
    write_table(suite.summary, pd.DataFrame(summary, columns=list(SUMMARY_COLUMNS)))
    comparison = comparison_rows(runs, paired, ensembles, base)
    write_table(suite.comparison, pd.DataFrame(comparison, columns=list(COMPARISON_COLUMNS)))
    validation = paired["validation"]
    return {
        "summary": str(suite.summary),
        "comparison": str(suite.comparison),
        "unpaired_accounts": validation.unpaired if validation is not None else None,
    }
