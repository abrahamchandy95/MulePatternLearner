"""mule audit on the loaded world (docs/design.md, "The pipeline", "Gates" and "Commands"), and
the one reading of its pass records that the mule train command makes.

The order of mule audit:

1. read_world through the fetch executor, matched to a configured world by its identity file
   (data.worlds);
2. the answer key through the evaluation executor, and read_labels through the fetch executor,
   into the evaluation store (evaluation.store);
3. PhantomLedger's records of the world (evaluation.gate_records);
4. when config scores the test world by its fallback and the world plays a test role, the
   fallback list (evaluation.fallback), into the store, and its rows fetched;
5. the label checks and the identification horizon (evaluation.label_checks), on every scored
   week's population (data.population, from the cache or listed);
6. the audit sample (evaluation.sample), its rows fetched, and the clock gain test
   (evaluation.clock_gain);
7. when the world plays a test role, the sample stored with the start of each row's week, and
   its rows fetched at those starts (evaluation.clock_invariance), which mule evaluate scores at
   both times;
8. on a development world, the feature audit (evaluation.feature_audit) and the development
   checks (evaluation.development_checks);
9. the pass record into the store: pass or fail per check, the query versions, the cache key,
   the hash of the settings and the world's name, kind and identity, and nothing else.

Every query runs through an entry point that allows it: read_answer_key through the
evaluation executor alone, everything else through the fetch executor; fetch refuses query
versions without a parity record. The report tables go to the store beside the pass record.

train_may_run is what the mule train command reads before any training code runs, the one
exception the owner agreed on 2026-10-17 (docs/design.md, "Gates"): the pass or fail of each
audit record of its rotation's worlds, and nothing else in them. A failure stops the run when
it is any failed check of a reported world, or a failed label check or feature audit of a
development world; a world without a pass record stops it too, since the operator runs mule
audit on each world before mule train. It returns the stopping checks by name, never a value.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from ..config import DEFAULT_CONFIG, RunConfig, WorldConfig
from ..data.labels import read_identified
from ..data.population import population
from ..data.worlds import TEST, find_world, roles
from ..paths import pl_gate_record
from ..subgraph.build import build, read_rows
from ..subgraph.cache import Cache
from ..subgraph.fetch import Fetcher, fetch
from ..tigergraph.executor import FetchExecutor, QueryExecutor
from ..tigergraph.queries import VERSIONS
from ..tigergraph.world import cache_key, canonical_hash
from . import clock_gain, clock_invariance, development_checks, feature_audit, label_checks
from .answer_key import read_answer_key
from .fallback import by_week, fallback_keys
from .gate_records import check_records
from .sample import audit_sample
from .store import AuditRecord, Store

GATE_RECORD = "gate_record"
CLOCK_GAIN = "clock_gain"
FEATURE_AUDIT = "feature_audit"
NO_POSITIVE_HUB = "no_positive_hub"
# The kind of world on which the development checks run.
DEVELOPMENT = "development"
REPORTED = "reported"
# The examples of the audit sample built into one batch at a time.
BATCH = 512


def audit_settings(config: RunConfig) -> str:
    """The hash of the settings an audit's checks were made under: the audit's own, the
    calendar, the data settings (the draw key among them) and the subgraph settings."""
    return canonical_hash(
        {
            "audit": asdict(config.audit),
            "calendar": {
                **asdict(config.calendar),
                "export_start": config.calendar.export_start.isoformat(),
            },
            "data": {**asdict(config.data), "draw_key": config.data.draw_key.hex()},
            "subgraph": asdict(config.subgraph),
        }
    )


@dataclass
class Measured:
    """What the audit reads of each account-week of its sample, in the sample's order: its
    controls and clock values for the gain test, and on a development world its audited columns,
    its own devices and IPs, the customer that owns it, its short dwell shares with
    left_by_card and its out_30d."""

    controls: list[NDArray[np.float64]] = field(default_factory=lambda: list[NDArray[np.float64]]())
    clock: list[NDArray[np.float64]] = field(default_factory=lambda: list[NDArray[np.float64]]())
    columns: list[NDArray[np.float64]] = field(default_factory=lambda: list[NDArray[np.float64]]())
    nodes: list[development_checks.OwnNodes] = field(
        default_factory=lambda: list[development_checks.OwnNodes]()
    )
    customers: list[str | None] = field(default_factory=lambda: list[str | None]())
    shares: list[list[float]] = field(default_factory=lambda: list[list[float]]())
    out_30d: list[float] = field(default_factory=lambda: list[float]())


def measure(cache: Cache, world: str, sample: pd.DataFrame, development: bool) -> Measured:
    """Read every account-week of the sample from the cache, a week at a time, in the
    sample's order."""
    found = Measured()
    t_values = sample["t_ms"].tolist()
    if t_values != sorted(t_values):
        raise ValueError("The audit sample is not in the order of its weeks")
    for t_ms, accounts in by_week(sample).items():
        rows = read_rows(cache, [(t_ms, account) for account in accounts])
        for example in rows:
            events = clock_gain.durations(example)
            found.controls.append(
                clock_gain.control_values(events, clock_gain.node_counts(example))
            )
            found.clock.append(clock_gain.clock_values(t_ms, events))
        if not development:
            continue
        for start in range(0, len(rows), BATCH):
            batch = build(rows[start : start + BATCH], world)
            found.columns += list(feature_audit.audit_columns(batch))
        for example in rows:
            found.nodes.append(development_checks.own_nodes(example))
            found.customers.append(development_checks.customer_of(example))
            shares, out_30d = development_checks.flow_shares(example)
            found.shares.append(shares)
            found.out_30d.append(out_30d)
    return found


@dataclass(frozen=True)
class Audited:
    """What mule audit found: the pass record it wrote, and why each failed check failed, by
    name (the gate record's failures, or the accounts breaking a label check), for the
    operator; the report tables are in the store."""

    record: AuditRecord
    reasons: Mapping[str, tuple[str, ...]]


def _matrix(rows: Sequence[NDArray[np.float64]], width: int) -> NDArray[np.float64]:
    return np.asarray(rows, dtype=np.float64).reshape(len(rows), width)


def audit(
    executor: QueryExecutor,
    config: RunConfig = DEFAULT_CONFIG,
    *,
    identity_files: Sequence[Path] | None = None,
    gate_file: Path | None = None,
    parity: Path | None = None,
) -> Audited:
    """Audit the loaded world, as the module says. identity_files are the identity files to
    match it among and gate_file its gate record, by default those in the records folder .env
    names; parity is the parity record fetch reads, by default gsql/parity.toml."""
    fetching = FetchExecutor(executor)
    loaded = find_world(fetching, config.worlds, identity_files)
    store = Store(loaded.identity)
    key = read_answer_key(executor)
    store.write_answer_key(key)
    labels = read_identified(fetching)
    store.write_labels(labels)
    checks: dict[str, bool] = {}
    reasons: dict[str, tuple[str, ...]] = {}

    gates = check_records(
        loaded.world, loaded.path, pl_gate_record(loaded.seed) if gate_file is None else gate_file
    )
    checks[GATE_RECORD] = not gates
    reasons[GATE_RECORD] = gates

    cache = Cache.of(loaded.identity, config.subgraph)
    tested = TEST in roles(loaded.config.name, config.worlds.rotations)
    scoring_times = config.calendar.scoring_times(loaded.config.seed)
    found = {t_ms: population(fetching, cache.root, t_ms) for t_ms in scoring_times}
    populations = {t_ms: [row.account_id for row in rows] for t_ms, rows in found.items()}
    if config.data.scoring == "fallback" and tested:
        keys = fallback_keys(key, populations, config.data, scoring_times)
        store.write_fallback(keys)
        fetch(executor, cache, by_week(keys), config.transport, parity)

    labelled = label_checks.check_labels(
        key, labels.accounts, populations, scoring_times, config.audit.horizon_level
    )
    checks |= labelled.results
    reasons |= {name: labelled.breaking[name] for name in label_checks.LABEL_CHECKS}
    store.write_table("label_counts", labelled.counts)
    store.write_table("mules_never_scored", labelled.unscored)
    store.write_table("identification_horizon", labelled.horizon)

    sample = audit_sample(
        key, populations, config.audit.nonmule_sample, config.data.draw_key, scoring_times
    )
    fetch(executor, cache, by_week(sample), config.transport, parity)
    if tested:
        starts = dict(zip(scoring_times, config.calendar.week_starts(), strict=True))
        scorable = {
            start: [row.account_id for row in population(fetching, cache.root, start)]
            for start in sorted({starts[int(t_ms)] for t_ms in sample["t_ms"].tolist()})
        }
        stored = clock_invariance.with_week_starts(sample, starts, scorable)
        store.write_clock_sample(stored)
        fetch(
            executor,
            cache,
            by_week(clock_invariance.week_start_keys(stored)),
            config.transport,
            parity,
        )
    development = loaded.config.kind == DEVELOPMENT
    measured = measure(cache, loaded.identity, sample, development)
    positive = sample["positive"].to_numpy(dtype=bool)
    weight = sample["weight"].to_numpy(dtype=np.float64)
    clusters = sample["cluster"].tolist()
    seed = config.training.bootstrap_seed
    gained = clock_gain.gain(
        _matrix(measured.controls, clock_gain.CONTROLS),
        _matrix(measured.clock, clock_gain.WIDTH),
        positive,
        weight,
        clusters,
        config.audit,
        seed,
    )
    checks[CLOCK_GAIN] = gained.passed
    store.write_table(CLOCK_GAIN, gained.frame())

    if development:
        audited = feature_audit.feature_audit(
            _matrix(measured.columns, len(feature_audit.COLUMNS)),
            positive,
            weight,
            clusters,
            config.audit,
            seed,
        )
        checks[FEATURE_AUDIT] = audited.passed
        reasons[FEATURE_AUDIT] = audited.stopping
        store.write_table("feature_audit", audited.table)
        fetcher = Fetcher(executor, cache, config.transport)
        weeks = sorted(set(sample.loc[positive, "t_ms"].tolist()))
        hubs = {
            int(t_ms): frozenset(hub.account_id for hub in fetcher.hubs(int(t_ms)).accounts)
            for t_ms in weeks
        }
        mules = sample[positive]
        hub_rows = development_checks.positive_hubs(
            list(zip(mules["t_ms"].tolist(), mules["account_id"].tolist(), strict=True)), hubs
        )
        checks[NO_POSITIVE_HUB] = hub_rows.empty
        store.write_table("positive_hubs", hub_rows)
        chosen = np.flatnonzero(positive)
        active = [
            development_checks.ActiveMule(int(t_ms), str(account), measured.customers[index], ring)
            for index, t_ms, account, ring in zip(
                chosen.tolist(),
                mules["t_ms"].tolist(),
                mules["account_id"].tolist(),
                mules["ring"].tolist(),
                strict=True,
            )
        ]
        store.write_table(
            "ring_leaf_share",
            development_checks.ring_leaf_share(active, [measured.nodes[index] for index in chosen]),
        )
        store.write_table(
            "card_spending",
            development_checks.card_spending(
                np.asarray(measured.shares, dtype=np.float64).reshape(len(measured.shares), 5),
                np.asarray(measured.out_30d, dtype=np.float64),
                positive,
                weight,
            ),
        )

    record = AuditRecord(
        loaded.config.name,
        loaded.config.kind,
        loaded.identity,
        cache_key(config.subgraph),
        audit_settings(config),
        dict(VERSIONS),
        checks,
    )
    store.write_record(record)
    return Audited(record, {name: reasons.get(name, ()) for name in record.failed})


def train_may_run(worlds: Sequence[tuple[WorldConfig, str]]) -> tuple[str, ...]:
    """The checks of the audit records of a rotation's worlds, each given with its identity,
    that stop mule train, as "<world>: <check>"; empty when training may run."""
    stops: list[str] = []
    for world, identity in worlds:
        record = Store(identity).record()
        if record is None:
            stops.append(f"{world.name}: no audit record")
            continue
        if record.world != world.name:
            stops.append(f"{world.name}: the audit record is of the world {record.world}")
            continue
        for check in record.failed:
            if world.kind == REPORTED or check.startswith("label.") or check == FEATURE_AUDIT:
                stops.append(f"{world.name}: {check}")
    return tuple(stops)
