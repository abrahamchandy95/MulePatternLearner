"""mule evaluate's results (docs/design.md, "Gates", "Evaluation" and "Results").

A result is reported only when, for each of its rotation's three worlds, the world is a
reported one and its pass record (evaluation.store) shows PhantomLedger's records passed, every
label check, the identification horizon and the clock probe passed, under the current query
versions and settings; and when every world of the development rotation has a passing feature
audit of the same query versions and settings (verdict). Otherwise every table is marked
development only, as on development worlds. Nothing here reads a pass record's values: there
are none.

evaluate scores nothing itself: it takes each run's logits of the test keys (evaluation.rankings
scores the model files) and measures the runs one at a time, from a test frame built for each and
dropped, so that memory holds one run's frame beside the selected run's (a frame of a reported
world is 23 million rows, about 2.6 GB). It writes, into the run's results folder, a table per
kind, each with the rotation's name and a development_only column:

- metrics: per ranker and seed, the headline metrics of the main ranking and the others of
  evaluation.rankings, and for the model mule score uses (the GNN seed with the best selection
  score, marked selected) each headline metric's ring-resampled interval;
- per_monday: recall and precision at each budget per Monday, per ranker and seed;
- groups: each group's metrics (evaluation.groups), per ranker and seed;
- spread: per ranker, group and metric, the mean, standard deviation, least and greatest over
  the seeds (and, once every rotation is evaluated, over the rotations: rotation_spread);
- comparisons: per ranker against the GNN, the difference of the seeds' mean of each headline
  metric (the GNN's minus the ranker's) with its paired, ring-resampled interval; shown only
  when the interval excludes 0, inconclusive otherwise;
- mules: the selected model's table of true mules (evaluation.rankings.mule_table);
- the missed-mule report (evaluation.missed) of the selected model;
- identification bias (evaluation.bias): victim_origin, origin_by_identification,
  hidden_recall_by_origin, identification_separation, and on the development rotation the
  selection_epochs and selection_correlation the caller rescored.

The manifest records the code commit and whether the tree was dirty (a reported result refuses
a dirty tree), the query versions, the world identities, the GPU name and count, the CUDA,
torch, PyG and XGBoost versions, wall time per stage, TigerGraph calls and cache hits, what each
tuned setting was chosen from and which earlier runs' reports informed a change, whether the
result is reported and why not. Every file is written through a temporary file and a rename,
and each command appends to events.jsonl.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import pyarrow as pa
import torch
import torch_geometric
import xgboost

from ..config import AuditConfig, WorldConfig
from ..subgraph.cache import Cache
from ..tigergraph.queries import VERSIONS
from . import bias, groups, missed
from .answer_key import ACTIVE, EARLY_WARNING, AnswerKey
from .audit import CLOCK_PROBE, DEVELOPMENT, FEATURE_AUDIT, GATE_RECORD, REPORTED
from .hops import hop_distances
from .label_checks import IDENTIFICATION_HORIZON, LABEL_CHECKS
from .rankings import HEADLINE, evaluation_frame, ring_links, run_metrics, scored_keys
from .store import Store, write_parquet
from .uncertainty import Clusters, cluster_keys, percentiles, replicate_matrix, spread

# The checks a reported world's pass record must show passed.
REPORTED_CHECKS = (GATE_RECORD, *LABEL_CHECKS, IDENTIFICATION_HORIZON, CLOCK_PROBE)
# The ranker the comparisons are made against, and whose best seed mule score uses.
GNN = "gnn"
EVENTS_FILE = "events.jsonl"
MANIFEST_FILE = "manifest.json"
DEVELOPMENT_ONLY = "development_only"


@dataclass(frozen=True)
class Verdict:
    """Why a result may not be reported, by name; reported when there is no reason."""

    reasons: tuple[str, ...]

    @property
    def reported(self) -> bool:
        return not self.reasons


@dataclass(frozen=True)
class MadeUnder:
    """What a pass record must have been made under to count: every query version, the cache
    key and the hash of the audit's settings (evaluation.audit.audit_settings)."""

    versions: Mapping[str, str]
    cache_key: str
    settings: str


def verdict(
    rotation: Sequence[tuple[WorldConfig, str]],
    development: Sequence[tuple[WorldConfig, str]],
    current: MadeUnder,
) -> Verdict:
    """Whether a rotation's result may be reported: rotation gives its three worlds with their
    identities, development those of the development rotation, and current what their pass
    records must have been made under."""
    reasons: list[str] = []

    def made_under(world: WorldConfig, identity: str) -> dict[str, bool] | None:
        record = Store(identity).record()
        if record is None:
            reasons.append(f"{world.name} has no audit record")
            return None
        if record.world != world.name:
            reasons.append(f"{world.name}'s audit record is of the world {record.world}")
            return None
        if (dict(record.versions), record.cache_key, record.settings) != (
            dict(current.versions),
            current.cache_key,
            current.settings,
        ):
            reasons.append(f"{world.name} was audited under other query versions or settings")
            return None
        return dict(record.checks)

    for world, identity in rotation:
        if world.kind != REPORTED:
            reasons.append(f"{world.name} is a {world.kind} world")
        checks = made_under(world, identity)
        if checks is not None:
            reasons += [
                f"{world.name}: {check} did not pass"
                for check in REPORTED_CHECKS
                if not checks.get(check, False)
            ]
    if not development:
        reasons.append("no world of the development rotation is given")
    for world, identity in development:
        if world.kind != DEVELOPMENT:
            reasons.append(f"{world.name} is not a development world")
        checks = made_under(world, identity)
        if checks is not None and not checks.get(FEATURE_AUDIT, False):
            reasons.append(f"{world.name}: the feature audit did not pass")
    return Verdict(tuple(reasons))


@dataclass(frozen=True)
class Run:
    """A ranker's seed and its logits of the test keys (t_ms, account_id, logit), with the
    selection score it was chosen by."""

    ranker: str
    seed: int
    selection_score: float
    scores: pd.DataFrame


@dataclass(frozen=True)
class ScoredWorld:
    """The test world as evaluation reads it: its identity and cache, the keys scored with
    their weights (t_ms, account_id, weight), its answer key and its identified mules."""

    identity: str
    cache: Cache
    keys: pd.DataFrame
    key: AnswerKey
    identified: frozenset[str]

    @classmethod
    def of(cls, identity: str, cache: Cache, keys: pd.DataFrame) -> ScoredWorld:
        """The test world of an identity, its answer key and labels read from its evaluation
        store, which mule audit filled while the world was loaded."""
        store = Store(identity)
        return cls(identity, cache, keys, store.answer_key(), store.labels().accounts)


def selected(runs: Sequence[Run]) -> Run:
    """The GNN run mule score uses: the best selection score, the lowest seed among equals."""
    full = [run for run in runs if run.ranker == GNN]
    if not full:
        raise ValueError("Evaluation needs a run of the GNN")
    return min(full, key=lambda run: (-run.selection_score, run.seed))


def _long(
    ranker: str, seed: int, group: str, values: Mapping[str, float]
) -> list[dict[str, object]]:
    return [
        {"ranker": ranker, "seed": seed, "group": group, "metric": name, "value": float(value)}
        for name, value in values.items()
    ]


@dataclass(frozen=True)
class _Measured:
    """What one run's measurement leaves, small enough to keep for every run at once: its
    metrics by name, its per-Monday table, its group table and its headline metrics on each
    replicate of the ring bootstrap (replicates by HEADLINE)."""

    values: Mapping[str, float]
    per_monday: pd.DataFrame
    groups: pd.DataFrame
    replicates: NDArray[np.float64]


def evaluate(
    world: ScoredWorld,
    runs: Sequence[Run],
    audit: AuditConfig,
    seed: int,
    folder: Path,
    *,
    rotation: str,
    verdict_found: Verdict,
    epochs: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Evaluate every run on the test world and write the results into folder, as the module
    says; seed is the seed of the ring bootstrap's draws, and epochs, on the development
    rotation, the selection world's rescored epochs (bias.rescore_epochs). The tables written,
    by name.

    A test frame holds a row for every scored account-Monday, with its strings, 23 million rows
    and about 2.6 GB for a reported world: so the runs are measured one at a time, each from a
    frame built for it and dropped, and what is kept of each is small (_Measured). Only the
    selected run's frame is kept, since the missed-mule report, the groups and the identification
    bias read it, and it is measured first: the ring bootstrap resamples its clusters, which every
    run shares, since every run ranks the same account-Mondays."""
    if len({(run.ranker, run.seed) for run in runs}) != len(runs):
        raise ValueError("Two runs are of one ranker and seed")
    best = selected(runs)

    def frame_of(run: Run) -> pd.DataFrame:
        return evaluation_frame(run.scores, world.keys, world.key, world.identified)

    frame = frame_of(best)
    links = ring_links(world.cache, frame, scored_keys(frame, (EARLY_WARNING,)))
    quiet = groups.quiet_rows(world.cache, scored_keys(frame, (ACTIVE, EARLY_WARNING)))
    hops = hop_distances(world.cache, scored_keys(frame, (ACTIVE,)), world.identified)
    surfaced = groups.surfaced_rings(world.key, world.identified)

    measured: dict[tuple[str, int], _Measured] = {}
    found_mules: pd.DataFrame | None = None
    rows_of_groups: dict[str, NDArray[np.bool_]] = {}
    clusters: Clusters | None = None
    reference: tuple[NDArray[np.int64], NDArray[np.bool_], NDArray[np.float64]] | None = None
    for run in [best, *(other for other in runs if other is not best)]:
        run_frame = frame if run is best else frame_of(run)
        found = run_metrics(run_frame, links)
        ranking = found.ranking
        rows = (
            ranking.frame["t_ms"].to_numpy(dtype=np.int64),
            ranking.positive,
            ranking.weight,
        )
        if clusters is None or reference is None:
            clusters = Clusters.of(
                cluster_keys(
                    ranking.frame["account_id"].tolist(),
                    ranking.positive.astype(np.int64),
                    ranking.frame["mule_ring_id"].to_numpy(dtype=np.int64),
                ),
                ranking.positive,
            )
            reference = rows
        elif any(
            not np.array_equal(held, same) for held, same in zip(rows, reference, strict=True)
        ):
            raise ValueError("The runs did not rank the same account-Mondays")
        chosen = groups.group_rows(run_frame, surfaced, quiet, hops)
        measured[(run.ranker, run.seed)] = _Measured(
            found.values,
            found.per_monday.assign(ranker=run.ranker, seed=run.seed),
            groups.group_metrics(run_frame, chosen).assign(ranker=run.ranker, seed=run.seed),
            replicate_matrix(
                [ranking.headline],
                clusters,
                reference[2],
                replicates=audit.replicates,
                seed=seed,
            )[:, 0, :],
        )
        if run is best:
            found_mules = found.mules
            rows_of_groups = chosen
        del run_frame, found, ranking, rows, chosen
    assert found_mules is not None

    metrics_rows: list[dict[str, object]] = []
    for run in runs:
        metrics_rows += _long(run.ranker, run.seed, "all", measured[(run.ranker, run.seed)].values)
    metrics = pd.DataFrame(metrics_rows)
    metrics["selected"] = (metrics["ranker"] == best.ranker) & (metrics["seed"] == best.seed)
    metrics["low"] = np.nan
    metrics["high"] = np.nan
    low, high = percentiles(measured[(best.ranker, best.seed)].replicates, audit.interval)
    for position, name in enumerate(HEADLINE):
        chosen_rows = (
            metrics["selected"] & (metrics["group"] == "all") & (metrics["metric"] == name)
        )
        metrics.loc[chosen_rows, "low"] = low[position]
        metrics.loc[chosen_rows, "high"] = high[position]

    def headline(run: tuple[str, int]) -> NDArray[np.float64]:
        return np.array([measured[run].values[name] for name in HEADLINE])

    comparisons: list[dict[str, object]] = []
    gnn = [(run.ranker, run.seed) for run in runs if run.ranker == GNN]
    for ranker in sorted({run.ranker for run in runs} - {GNN}):
        other = [(run.ranker, run.seed) for run in runs if run.ranker == ranker]
        differences = np.mean([measured[name].replicates for name in gnn], axis=0) - np.mean(
            [measured[name].replicates for name in other], axis=0
        )
        point = np.mean([headline(name) for name in gnn], axis=0) - np.mean(
            [headline(name) for name in other], axis=0
        )
        d_low, d_high = percentiles(differences, audit.interval)
        for position, name in enumerate(HEADLINE):
            shown = bool(
                np.isfinite(d_low[position]) and (d_low[position] > 0 or d_high[position] < 0)
            )
            comparisons.append(
                {
                    "ranker": ranker,
                    "metric": name,
                    "difference": float(point[position]),
                    "low": float(d_low[position]),
                    "high": float(d_high[position]),
                    "shown": shown,
                }
            )

    every_group = pd.concat(
        [measured[(run.ranker, run.seed)].groups for run in runs], ignore_index=True
    )
    combined = pd.concat(
        [metrics.loc[:, ["ranker", "seed", "group", "metric", "value"]], every_group],
        ignore_index=True,
    )
    spread_rows = [
        {"ranker": ranker, "group": group, "metric": metric, **spread(held["value"].tolist())}
        for (ranker, group, metric), held in combined.groupby(["ranker", "group", "metric"])
    ]
    mules = found_mules.copy()
    names = groups.group_names(frame, rows_of_groups)
    mules["groups"] = [names.get(account, "") for account in mules["account_id"]]
    origin = bias.victim_origin(world.cache, frame)
    tables: dict[str, pd.DataFrame] = {
        "metrics": metrics,
        "per_monday": pd.concat(
            [measured[(run.ranker, run.seed)].per_monday for run in runs], ignore_index=True
        ),
        "groups": every_group,
        "spread": pd.DataFrame(spread_rows),
        "comparisons": pd.DataFrame(
            comparisons, columns=["ranker", "metric", "difference", "low", "high", "shown"]
        ),
        "mules": mules,
        "victim_origin": origin,
        "origin_by_identification": bias.origin_table(origin),
        "hidden_recall_by_origin": bias.hidden_recall_by_origin(origin, found_mules),
        "identification_separation": bias.identification_separation(
            world.cache, frame, world.identity
        ),
    }
    if epochs is not None:
        tables["selection_epochs"] = epochs
        tables["selection_correlation"] = bias.selection_correlation(epochs)
    development_only = not verdict_found.reported
    write_tables(folder, tables, rotation, development_only)
    reported = missed.missed_mules(found_mules, audit.report_budget)
    missed.write_report(
        folder,
        world.cache,
        world.identity,
        frame,
        reported,
        rows_of_groups,
        {"rotation": rotation, DEVELOPMENT_ONLY: development_only},
    )
    return tables


def rotation_spread(metrics: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """The spread over the rotations of each ranker's metrics, from the metrics table each
    rotation's evaluation wrote, by rotation: per ranker, group and metric, the spread of the
    seeds' mean, and of the selected model's value for the GNN."""
    rows: list[dict[str, object]] = []
    combined = pd.concat(
        [table.assign(rotation=name) for name, table in metrics.items()], ignore_index=True
    )
    for (ranker, group, metric), held in combined.groupby(["ranker", "group", "metric"]):
        means = held.groupby("rotation")["value"].mean()
        rows.append(
            {
                "ranker": ranker,
                "group": group,
                "metric": metric,
                "of": "seed_mean",
                "rotations": len(means),
                **spread(means.tolist()),
            }
        )
        chosen = held[held["selected"]]
        if not chosen.empty:
            rows.append(
                {
                    "ranker": ranker,
                    "group": group,
                    "metric": metric,
                    "of": "selected",
                    "rotations": int(chosen["rotation"].nunique()),
                    **spread(chosen["value"].tolist()),
                }
            )
    return pd.DataFrame(rows)


def write_tables(
    folder: Path, tables: Mapping[str, pd.DataFrame], rotation: str, development_only: bool
) -> None:
    """Write each table as <name>.parquet, with the rotation's name and whether it is
    development only in every row."""
    for name, table in tables.items():
        marked = table.assign(rotation=rotation, **{DEVELOPMENT_ONLY: development_only})
        write_parquet(
            folder / f"{name}.parquet", pa.Table.from_pandas(marked, preserve_index=False)
        )


def git_state(where: Path | None = None) -> tuple[str, bool]:
    """The commit the code runs from and whether its tree differs from it, asked of git in the
    package's own folder, which lies in the repository."""
    folder = Path(__file__).resolve().parent if where is None else where
    commit = subprocess.run(
        ["git", "-C", str(folder), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(folder), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return commit, bool(status.strip())


def manifest(
    *,
    verdict_found: Verdict,
    identities: Mapping[str, str],
    wall_seconds: Mapping[str, float],
    tigergraph_calls: int,
    cache_hits: int,
    chosen_from: Mapping[str, str],
    informed_by: Sequence[str],
    git: tuple[str, bool] | None = None,
) -> dict[str, object]:
    """The manifest of a run's results. A reported result refuses a dirty tree."""
    commit, dirty = git_state() if git is None else git
    if verdict_found.reported and dirty:
        raise ValueError(
            f"A reported result needs a clean tree, and the tree differs from {commit}: commit "
            "the code, or the result is development only"
        )
    devices = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    return {
        "created": datetime.now(UTC).isoformat(),
        "commit": commit,
        "dirty": dirty,
        "reported": verdict_found.reported,
        "not_reported_because": list(verdict_found.reasons),
        "versions": dict(VERSIONS),
        "world_identities": dict(identities),
        "gpus": devices,
        "gpu_count": len(devices),
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "xgboost": xgboost.__version__,
        "wall_seconds": dict(wall_seconds),
        "tigergraph_calls": tigergraph_calls,
        "cache_hits": cache_hits,
        "chosen_from": dict(chosen_from),
        "informed_by": list(informed_by),
    }


def write_manifest(folder: Path, content: Mapping[str, object]) -> None:
    """Write the manifest through a temporary file and a rename."""
    text = json.dumps(content, indent=2, sort_keys=True)
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / f".{MANIFEST_FILE}.tmp"
    try:
        temporary.write_text(text)
        temporary.replace(folder / MANIFEST_FILE)
    finally:
        temporary.unlink(missing_ok=True)


def append_event(
    folder: Path, command: str, event: str, error: BaseException | None = None
) -> None:
    """Append a line to the run's events.jsonl: when, which command, what happened, and the
    error that stopped it."""
    folder.mkdir(parents=True, exist_ok=True)
    line = {
        "time": datetime.now(UTC).isoformat(),
        "command": command,
        "event": event,
        "error": None if error is None else f"{type(error).__name__}: {error}",
    }
    with (folder / EVENTS_FILE).open("a") as events:
        events.write(json.dumps(line) + "\n")
