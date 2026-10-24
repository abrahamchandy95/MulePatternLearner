"""mule bench: what the loaded world costs, and what that projects to at other sizes (docs/design.md,
"Scaling and the benchmark").

It measures a sample of the loaded world and writes one long table to results/bench/<world>.parquet:
the columns section, name, variant, value, unit and note, a row per number, which a person can
open and filter by hand. The sample is a few scored Mondays (bench.mondays: the first, the last
and those between), and on each the accounts of a training draw (data.examples), unlabelled
accounts and the identified active mules, reported apart. The table holds, by section:

- world: the loaded world, TigerGraph's version, its counts of vertices and edges and the seconds
  read_world took (TigerGraph's memory in use is read in the Cloud console, since the metrics
  endpoint answers 403 for this workspace's secret);
- fetch: per query and request, at each of bench.concurrency concurrent requests, the seconds,
  the request and response bytes and the work counters; B1's worst account and its steps;
- subgraph and cache: the sizes of the fetched subgraphs by type, the community sizes, and the
  cache's bytes per account and in all;
- build and gpu: the seconds to read a built example's rows and build its batch, the forward
  pass and a training step on the device, the peak memory and the examples a second;
- handoff: the bytes the parent writes under TMPDIR for the XGBoost process
  (comparisons.process), and the seconds to write, read and score a matrix;
- hubs and profile: profile_hubs' histograms, the hubs and crowds against the thresholds, and
  the count and share of every Monday's identified active mules (all of them, not only the ones
  fetched) that are account hubs, which must be 0;
- install: the seconds to install each query again, with INSTALL QUERY -FORCE;
- co_users: what the choice between co-user rows per Monday for every customer and on demand
  needs; the rule is the same either way;
- selection, audit, bootstrap and scoring: the projections the owner asked for (below).

The projections. Each takes a measured rate and the counts of a world of that many people,
people times the loaded world's scorable accounts per person per Monday times the scored
Mondays, and never changes a design: it prices what the design says.

- selection (docs/design.md, "Training and selection"): the selection world's examples are built
  anew on every pass (training.batches), and a rotation makes passes_per_rotation passes
  (training.trainer.selection_passes: seeds x (priors + ablations) x epochs). The table gives the
  seconds to build one example, the examples a pass scores, the passes a rotation makes, with the
  prior grid and with the prior fixed, and the projected seconds a rotation spends on selection,
  for the build alone and with the forward pass, at each of bench.projected_people. When
  training.reuse_built_examples keeps the examples (the proof of concept), the first pass builds
  them and every pass runs only the forward pass: the build is priced once.
- audit (docs/design.md, "mule audit"): mule audit fetches every active true-mule account-Monday
  and audit.nonmule_sample non-mule account-Mondays on every world. The sparse sample's seconds
  per account-Monday are measured, and priced at the sample size plus the identified mules' active
  account-Mondays, a lower bound on the mules' share: the answer key is evaluation's alone, so
  the bench never reads the true mules.
- bootstrap (docs/design.md, "Uncertainty"): mule evaluate's paired bootstrap costs
  audit.replicates x runs x ranked account-Mondays. The seconds one replicate's headline metrics
  take per ranked row are measured on a synthetic ranking, and priced at the replicates, at the
  runs (training.seeds x (the full model and its ablations), and each XGBoost comparison once) and
  at the ranked account-Mondays of each size. The reported estimate of 35 runs came from XGBoost
  with five seeds; with each comparison trained once the runs are 27.
- scoring: the whole-world scoring run, against its fallback (data.scoring): the fetch it needs
  at the sparse rate (an upper bound, since a whole world's neighbours are shared), and one
  scoring pass of every run of the test world, which builds the batches anew for each, or once
  when training.reuse_built_examples keeps them.

The table is written through a temporary file and a rename.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import resource
import sys
import tempfile
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import pyarrow as pa
import torch
from torch.optim import Adam

from .comparisons import process
from .comparisons.features import GRAPH_COLUMNS, vectors_of
from .comparisons.rankers import COLUMNS as XGBOOST_RANKERS
from .comparisons.rounds import MAX_ROUNDS
from .config import DEFAULT_CONFIG, RunConfig
from .data.calendar import check_window
from .data.examples import monday, unlabelled
from .data.labels import Labels, read_identified
from .data.population import list_population
from .data.worlds import LoadedWorld, find_world
from .evaluation.atomic import write_parquet
from .evaluation.ranking import Ranking
from .evaluation.statuses import ACTIVE, NON_MULE
from .evaluation.uncertainty import Clusters, replicate_matrix
from .model.encode import encode_batch
from .model.hetero import HeteroModel
from .model.loss import NonNegativePULoss
from .paths import bench_dir
from .runtime.console import show
from .subgraph.build import Batch, account_nodes, build, read_rows
from .subgraph.cache import Cache, Table
from .subgraph.fetch import PARAMS, PER_ACCOUNT, Fetcher, chunks
from .subgraph.layouts import LAYOUT
from .tigergraph.executor import BenchExecutor, ConnectionExecutor
from .tigergraph.queries import (
    PROFILE_HUBS,
    QUERY_FILES,
    READ_HUBS,
    READ_MONEY_FLOW,
    profile_hubs,
    read_hubs,
)
from .tigergraph.queries.read_hubs import Hubs
from .tigergraph.queries.read_money_flow import MoneyFlows
from .tigergraph.queries.read_world import World
from .tigergraph.world import identity
from .training.batches import ScoringBatches
from .training.trainer import ablations, logits, selection_passes

COLUMNS = ("section", "name", "variant", "value", "unit", "note")
UNLABELLED = "unlabelled"
POSITIVES = "positives"
SECONDS_PER_DAY = 86_400.0
# The training steps timed after one that warms the device up.
TIMED_STEPS = 3
# The most seconds a forced reinstall of a query may take before the bench stops waiting.
INSTALL_SECONDS = 3_600.0
# A prior for the timed training step, which no result depends on.
STEP_PRIOR = 0.01
# The rows the hand-off's timing booster is grown on.
FIT_ROWS = 2_000


# The table.


@dataclass
class Measures:
    """The rows of the bench table, as they are measured."""

    rows: list[tuple[str, str, str, float, str, str]] = field(
        default_factory=lambda: list[tuple[str, str, str, float, str, str]]()
    )

    def add(
        self, section: str, name: str, variant: str, value: float, unit: str, note: str = ""
    ) -> None:
        self.rows.append((section, name, variant, float(value), unit, note))

    def frame(self) -> pd.DataFrame:
        found = pd.DataFrame(self.rows, columns=list(COLUMNS))
        return found.astype(
            {"section": "str", "name": "str", "variant": "str", "value": "float64", "unit": "str"}
        ).astype({"note": "str"})

    def spread(
        self, section: str, name: str, variant: str, values: Sequence[float], unit: str
    ) -> None:
        """The mean, 90th percentile and greatest of values as three rows."""
        if not len(values):
            return
        held = np.asarray(values, dtype=np.float64)
        self.add(section, f"{name}_mean", variant, float(held.mean()), unit)
        self.add(section, f"{name}_p90", variant, float(np.quantile(held, 0.9)), unit)
        self.add(section, f"{name}_max", variant, float(held.max()), unit)


# The projections, as plain functions of what was measured.


def chosen_mondays(mondays: Sequence[int], count: int) -> tuple[int, ...]:
    """count of the Mondays, evenly spread, the first and the last among them: all of them when
    there are no more than count."""
    if len(mondays) <= count:
        return tuple(mondays)
    if count == 1:
        return (mondays[0],)
    last = len(mondays) - 1
    return tuple(mondays[round(index * last / (count - 1))] for index in range(count))


def examples_at(per_person: float, people: int, mondays: int) -> float:
    """The account-Mondays of a world of that many people: its scorable accounts per person on
    a Monday, over the scored Mondays."""
    return per_person * people * mondays


def runs_of(seeds: int, ablations: int, comparisons: int) -> int:
    """The runs mule evaluate measures: every seed of the full model and of each ablation, and
    each XGBoost comparison once."""
    return seeds * (1 + ablations) + comparisons


def seconds_projected(rate: float, count: float, passes: float = 1.0) -> float:
    """Seconds for count units at rate seconds each, over passes passes."""
    return rate * count * passes


def pass_seconds(
    build: float, forward: float, examples: float, passes: float, *, reuse: bool
) -> tuple[float, float]:
    """The seconds passes passes over a world's examples spend building them, and with the
    forward pass too: every pass builds every example anew, or, when reuse keeps the examples
    (training.batches.ScoringBatches), the first pass builds them and the others do not."""
    built = seconds_projected(build, examples, 1.0 if reuse else passes)
    return built, built + seconds_projected(forward, examples, passes)


# Measuring.


def _bytes(value: object) -> int:
    """The bytes of a request or an answer as compact JSON."""
    return len(json.dumps(value, separators=(",", ":")))


@dataclass(frozen=True)
class Request:
    """One request of a query: its Monday, its accounts and its parameters."""

    query: str
    t_ms: int
    accounts: tuple[str, ...]
    params: dict[str, Any]


@dataclass(frozen=True)
class Timed:
    """What a set of requests took at one concurrency: the wall seconds of all of them, each
    answered request's seconds, request bytes and response bytes, its answer (None for a request
    that failed, with the error in errors), so that one request TigerGraph refuses for its size
    is a number of the table and not the end of the bench."""

    wall: float
    seconds: list[float]
    sent: list[int]
    received: list[int]
    answers: list[list[dict[str, Any]] | None]
    errors: list[str]


def time_requests(executor: BenchExecutor, requests: Sequence[Request], concurrency: int) -> Timed:
    """Run the requests on concurrency threads, timing each and the whole."""

    def one(request: Request) -> tuple[float, int, int, list[dict[str, Any]] | None, str]:
        began = time.perf_counter()
        try:
            rows = executor.run(request.query, request.params)
        except Exception as error:
            failed = f"{type(error).__name__}: {str(error)[:200]}"
            return time.perf_counter() - began, _bytes(request.params), 0, None, failed
        return time.perf_counter() - began, _bytes(request.params), _bytes(rows), rows, ""

    began = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        done = list(pool.map(one, requests))
    wall = time.perf_counter() - began
    answered = [item for item in done if item[3] is not None]
    return Timed(
        wall,
        [item[0] for item in answered],
        [item[1] for item in answered],
        [item[2] for item in answered],
        [item[3] for item in done],
        [item[4] for item in done if item[4]],
    )


@dataclass(frozen=True)
class Sample:
    """The accounts measured on one Monday: how many are scorable, every identified active mule
    (identified), of which the first bench.positives are fetched (positives), the first of the
    unlabelled draw, which is fetched, and the Monday's hubs."""

    t_ms: int
    scorable: int
    identified: tuple[str, ...]
    positives: tuple[str, ...]
    unlabelled: tuple[str, ...]
    hubs: Hubs

    @property
    def identified_active(self) -> int:
        return len(self.identified)

    def group(self, name: str) -> tuple[str, ...]:
        return self.positives if name == POSITIVES else self.unlabelled


def take_samples(
    executor: BenchExecutor,
    mondays: Sequence[int],
    labels: Labels,
    config: RunConfig,
    measures: Measures,
) -> list[Sample]:
    """The sample of each chosen Monday, the hubs read and timed."""
    found: list[Sample] = []
    for t_ms in mondays:
        accounts = [row.account_id for row in list_population(executor, t_ms)]
        day = monday(t_ms, accounts, labels.mules)
        draw = unlabelled(day, labels.accounts, config.data)
        began = time.perf_counter()
        hubs = read_hubs.parse(executor.run(READ_HUBS, read_hubs.params(t_ms)))
        seconds = time.perf_counter() - began
        variant = f"t={t_ms}"
        measures.add("hubs", "read_hubs_seconds", variant, seconds, "s")
        found.append(
            Sample(
                t_ms,
                len(accounts),
                day.positives,
                day.positives[: config.bench.positives],
                draw.drawn[: config.bench.unlabelled],
                hubs,
            )
        )
    return found


def _requests(query: str, samples: Sequence[Sample], group: str, per_request: int) -> list[Request]:
    """The requests of a per-account query for a group of every sample: the Monday's hubs go
    with every one."""
    found: list[Request] = []
    for sample in samples:
        hubs = [hub.account_id for hub in sample.hubs.accounts]
        for chunk in chunks(sample.group(group), per_request):
            params = PARAMS[query](sample.t_ms, tuple(chunk), hubs)
            found.append(Request(query, sample.t_ms, tuple(chunk), params))
    return found


def measure_fetch(
    executor: BenchExecutor,
    samples: Sequence[Sample],
    config: RunConfig,
    measures: Measures,
) -> None:
    """Each per-account query's requests for each group, at each concurrency: seconds, bytes,
    and the work counters of the first run's answers."""
    per_request = config.transport.accounts_per_request
    for query, table in PER_ACCOUNT.items():
        for group in (UNLABELLED, POSITIVES):
            requests = _requests(query, samples, group, per_request)
            if not requests:
                continue
            for index, concurrency in enumerate(config.bench.concurrency):
                timed = time_requests(executor, requests, concurrency)
                variant = f"{group},concurrency={concurrency}"
                accounts = sum(len(request.accounts) for request in requests)
                measures.add("fetch", f"{query}.requests", variant, len(requests), "requests")
                measures.add(
                    "fetch",
                    f"{query}.requests_failed",
                    variant,
                    len(timed.errors),
                    "requests",
                    timed.errors[0] if timed.errors else "",
                )
                measures.add("fetch", f"{query}.accounts", variant, accounts, "accounts")
                measures.add("fetch", f"{query}.wall_seconds", variant, timed.wall, "s")
                measures.add(
                    "fetch",
                    f"{query}.accounts_per_second",
                    variant,
                    accounts / timed.wall if timed.wall > 0 else float("nan"),
                    "accounts/s",
                )
                measures.spread("fetch", f"{query}.request_seconds", variant, timed.seconds, "s")
                measures.spread("fetch", f"{query}.request_bytes", variant, timed.sent, "bytes")
                measures.spread(
                    "fetch", f"{query}.response_bytes", variant, timed.received, "bytes"
                )
                answered = [
                    request for request, rows in zip(requests, timed.answers, strict=True) if rows
                ]
                if index == 0 and answered:
                    per_account = [
                        size / len(request.accounts)
                        for size, request in zip(timed.received, answered, strict=True)
                    ]
                    measures.spread(
                        "fetch",
                        f"{query}.response_bytes_per_account",
                        variant,
                        per_account,
                        "bytes",
                    )
                    if query == READ_MONEY_FLOW:
                        _scan_steps(table, requests, timed, group, measures)


def _scan_steps(
    table: Table[Any],
    requests: Sequence[Request],
    timed: Timed,
    group: str,
    measures: Measures,
) -> None:
    """B1's work counter: the steps of each account's walk, its mean and its worst account."""
    steps: list[tuple[int, str]] = []
    for request, rows in zip(requests, timed.answers, strict=True):
        if rows is None:
            continue
        answer: MoneyFlows = table.parse(rows, request.accounts)
        steps += [(flow.scan_steps, flow.account_id) for flow in answer.accounts]
    if not steps:
        return
    values = [step for step, _ in steps]
    measures.spread("fetch", f"{READ_MONEY_FLOW}.scan_steps", group, values, "steps")
    worst, account = max(steps)
    measures.add("fetch", f"{READ_MONEY_FLOW}.worst_scan_steps", group, worst, "steps", account)


def fill_cache(
    executor: BenchExecutor,
    cache: Cache,
    samples: Sequence[Sample],
    config: RunConfig,
    measures: Measures,
) -> dict[str, list[tuple[int, str]]]:
    """Fetch the sample's rows into the bench's own cache, as mule fetch does and unlike the
    timed requests, which keep nothing, timing it. The keys of each group."""
    fetcher = Fetcher(executor, cache, config.transport)
    keys: dict[str, list[tuple[int, str]]] = {UNLABELLED: [], POSITIVES: []}
    for group in keys:
        began = time.perf_counter()
        accounts = 0
        for sample in samples:
            chosen = sample.group(group)
            if chosen:
                fetcher.monday(sample.t_ms, chosen)
                accounts += len(chosen)
            keys[group] += [(sample.t_ms, account) for account in chosen]
        seconds = time.perf_counter() - began
        if accounts:
            measures.add("fetch", "fill_seconds", group, seconds, "s")
            measures.add("fetch", "fill_accounts", group, accounts, "accounts")
            measures.add(
                "fetch",
                "fill_seconds_per_account_monday",
                group,
                seconds / accounts,
                "s",
                "B4 to B6 of each account and B1 to B3 of every account of its subgraph, as a sparse "
                "sample, which shares few neighbours",
            )
    return keys


def _size_of(folder: Path) -> int:
    return sum(path.stat().st_size for path in folder.rglob("*") if path.is_file())


def measure_subgraphs(
    cache: Cache, keys: dict[str, list[tuple[int, str]]], measures: Measures
) -> None:
    """The fetched subgraphs' sizes by type, the community sizes and the cache's bytes."""
    scored = 0
    for group, held in keys.items():
        if not held:
            continue
        scored += len(held)
        rows = read_rows(cache, held)
        sizes: dict[str, list[float]] = {
            "accounts": [],
            "payments": [],
            "customers": [],
            "devices": [],
            "ips": [],
            "enrolled_edges": [],
            "owner_enrolled_edges": [],
            "seen_edges": [],
            "tracked_inflows": [],
            "community_members": [],
        }
        for example in rows:
            sizes["accounts"].append(len(account_nodes(example.payments, example.identity)))
            sizes["payments"].append(len(example.payments.payments))
            sizes["customers"].append(len(example.identity.customers))
            sizes["devices"].append(len(example.identity.devices))
            sizes["ips"].append(len(example.identity.ips))
            sizes["enrolled_edges"].append(len(example.identity.enrolled))
            sizes["owner_enrolled_edges"].append(len(example.identity.owner_enrolled))
            sizes["seen_edges"].append(len(example.identity.seen))
            sizes["tracked_inflows"].append(len(example.inflows))
            sizes["community_members"].append(example.community.members)
        for name, values in sizes.items():
            measures.spread("subgraph", name, group, values, "nodes or edges")
        largest = max(range(len(rows)), key=lambda index: len(rows[index].payments.payments))
        measures.add(
            "subgraph",
            "largest_payments",
            group,
            len(rows[largest].payments.payments),
            "payments",
            held[largest][1],
        )
    total = _size_of(cache.root)
    measures.add("cache", "bytes_total", "sample", total, "bytes")
    if scored:
        measures.add("cache", "bytes_per_scored_account", "sample", total / scored, "bytes")


@dataclass(frozen=True)
class Rates:
    """The per-example seconds the projections price: building an example from its cached
    rows, and its forward pass."""

    build: float
    forward: float

    @property
    def scoring(self) -> float:
        return self.build + self.forward


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _peak_memory() -> float:
    """The most memory the process held: the device's peak when it is a GPU, else the process's
    peak resident set."""
    if torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated())
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(peak if sys.platform == "darwin" else peak * 1024)


def measure_model(
    cache: Cache,
    world: str,
    keys: dict[str, list[tuple[int, str]]],
    config: RunConfig,
    measures: Measures,
) -> tuple[Rates, list[Batch]]:
    """The seconds to read a built example's rows and build its batch, the forward pass, and a
    training step; with the device and its peak memory. The rates, and the scoring batches the
    XGBoost hand-off reads its matrix from."""
    every = [key for held in keys.values() for key in held]
    if not every:
        raise ValueError("The bench sample holds no account to build an example from")
    frame = pd.DataFrame(every, columns=["t_ms", "account_id"])
    batches = ScoringBatches(cache, world, frame, config.training.batch_size)
    built = list(batches)
    build_rate = batches.seconds / batches.built
    measures.add("build", "examples", "sample", batches.built, "examples")
    measures.add(
        "build",
        "seconds_per_example",
        "sample",
        build_rate,
        "s",
        "rows read from the cache and the batch built, measured on the sample: a pass reads the "
        "rows of up to 4,096 examples at a time, so the rate is approximate",
    )
    device = _device()
    torch.manual_seed(0)
    model = HeteroModel(LAYOUT, config.model).to(device)
    began = time.perf_counter()
    logits(model, built, None, device)
    forward_rate = (time.perf_counter() - began) / batches.built
    measures.add("build", "forward_seconds_per_example", "sample", forward_rate, "s", str(device))
    rows = read_rows(cache, every[: config.training.batch_size])
    labels = [int(index % 10 == 0) for index in range(len(rows))]
    batch = build(rows, world, labels=labels, weights=[1.0] * len(rows))
    loss = NonNegativePULoss(STEP_PRIOR)
    optimiser = Adam(model.parameters(), lr=config.training.learning_rate)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.train()
    seconds: list[float] = []
    for step in range(1 + TIMED_STEPS):
        began = time.perf_counter()
        data = encode_batch(batch.data, LAYOUT, device)
        optimiser.zero_grad()
        value, _ = loss(model(data), data.label, data.weight)
        value.backward()
        optimiser.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        if step:
            seconds.append(time.perf_counter() - began)
    step_seconds = float(np.mean(seconds))
    measures.add("gpu", "device", "", 1.0, "device", str(device))
    measures.add("gpu", "step_seconds", f"examples={len(rows)}", step_seconds, "s")
    measures.add("gpu", "training_examples_per_second", "", len(rows) / step_seconds, "examples/s")
    measures.add(
        "gpu", "scoring_examples_per_second", "", 1.0 / (build_rate + forward_rate), "examples/s"
    )
    measures.add("gpu", "peak_memory", "", _peak_memory(), "bytes", str(device))
    return Rates(build_rate, forward_rate), built


def measure_handoff(batches: Sequence[Batch], config: RunConfig, measures: Measures) -> float:
    """The cost of handing a matrix of the graph features to the XGBoost process
    (comparisons.process): the bytes written under TMPDIR, and the seconds to write it, read it
    back and score it through the process. A matrix of the sample's rows, repeated to
    bench.handoff_rows. The seconds to score a row, which the projections price."""
    vectors = vectors_of(batches)
    rows = config.bench.handoff_rows
    repeats = math.ceil(rows / len(vectors.graph))
    matrix: NDArray[np.float32] = np.tile(vectors.graph, (repeats, 1))[:rows]
    # The booster is a timing instrument: as many trees as the real ranker's, grown on the
    # first rows of the matrix with every tenth a positive.
    fitted = matrix[:FIT_ROWS]
    labels = (np.arange(len(fitted)) % 10 == 0).astype(np.int64)
    booster = process.fit(fitted, labels, np.ones(len(fitted)), GRAPH_COLUMNS)
    with tempfile.TemporaryDirectory(prefix="mule-bench-") as name:
        path = Path(name) / "x.npy"
        began = time.perf_counter()
        np.save(path, matrix, allow_pickle=False)
        write = time.perf_counter() - began
        written = path.stat().st_size
        began = time.perf_counter()
        np.load(path, allow_pickle=False)
        read = time.perf_counter() - began
    began = time.perf_counter()
    process.scores(booster, matrix, GRAPH_COLUMNS, MAX_ROUNDS)
    scored = time.perf_counter() - began
    variant = f"rows={len(matrix)}"
    measures.add("handoff", "bytes_written", variant, written, "bytes")
    measures.add("handoff", "bytes_per_row", variant, written / len(matrix), "bytes")
    measures.add("handoff", "write_seconds", variant, write, "s")
    measures.add("handoff", "read_seconds", variant, read, "s")
    measures.add(
        "handoff",
        "score_seconds",
        variant,
        scored,
        "s",
        "the whole call: write, start the process, read, score, write back, read back",
    )
    measures.add("handoff", "booster_bytes", "", len(booster), "bytes")
    return scored / len(matrix)


def measure_profile(
    executor: BenchExecutor, samples: Sequence[Sample], config: RunConfig, measures: Measures
) -> None:
    """profile_hubs' histograms, how many accounts, devices and IPs lie above the crowd and hub
    thresholds, and the share of the sample's identified active mules that are account hubs."""
    crowd, hub = config.subgraph.crowd_threshold, config.subgraph.hub_threshold
    for sample in samples:
        variant = f"t={sample.t_ms}"
        began = time.perf_counter()
        profile = profile_hubs.parse(executor.run(PROFILE_HUBS, profile_hubs.params(sample.t_ms)))
        measures.add("profile", "seconds", variant, time.perf_counter() - began, "s")
        for name, histogram in asdict(profile).items():
            for measure, count in histogram.items():
                measures.add("profile", name, f"{variant},customers={measure}", count, "nodes")
            measures.add(
                "profile",
                f"{name}_above_crowd",
                variant,
                sum(count for measure, count in histogram.items() if measure > crowd),
                "nodes",
            )
            measures.add(
                "profile",
                f"{name}_above_hub",
                variant,
                sum(count for measure, count in histogram.items() if measure > hub),
                "nodes",
            )
        hubs = {row.account_id for row in sample.hubs.accounts}
        measures.add("hubs", "account_hubs", variant, len(sample.hubs.accounts), "accounts")
        measures.add("hubs", "device_leaves", variant, len(sample.hubs.devices), "nodes")
        measures.add("hubs", "ip_leaves", variant, len(sample.hubs.ips), "nodes")
        active = sample.identified
        measures.add(
            "hubs",
            "identified_active_that_are_hubs",
            variant,
            sum(1 for account in active if account in hubs),
            "accounts",
            "must be 0",
        )
        measures.add(
            "hubs",
            "identified_active_hub_share",
            variant,
            sum(1 for account in active if account in hubs) / len(active) if active else 0.0,
            "share",
            "must be 0",
        )


def measure_bootstrap(config: RunConfig, measures: Measures) -> float:
    """The seconds one replicate of the paired bootstrap takes per ranked row: the headline
    metrics of a synthetic ranking of bench.bootstrap_rows account-Mondays over the scored
    Mondays, with mules in rings, resampled by cluster as mule evaluate resamples them
    (evaluation.uncertainty). The rate."""
    rows = config.bench.bootstrap_rows
    mondays = len(config.calendar.mondays())
    rng = np.random.default_rng(0)
    mules = max(2, rows // 1_000)
    status = np.full(rows, NON_MULE, dtype=object)
    status[:mules] = ACTIVE
    frame = pd.DataFrame(
        {
            "t_ms": np.arange(rows, dtype=np.int64) % mondays,
            "logit": rng.normal(size=rows) + 2.0 * (np.arange(rows) < mules),
            "weight": np.ones(rows),
            "status": status.astype(str),
        }
    )
    positive = (frame["status"] == ACTIVE).to_numpy()
    ranking = Ranking.of(frame, positive)
    accounts = np.arange(rows) // mondays
    keys = [
        f"ring:{index % max(1, mules // 4)}" if mule else f"account:{account}"
        for index, (mule, account) in enumerate(
            zip(positive.tolist(), accounts.tolist(), strict=True)
        )
    ]
    clusters = Clusters.of(keys, positive)
    replicates = config.bench.bootstrap_replicates
    began = time.perf_counter()
    replicate_matrix([ranking.headline], clusters, ranking.weight, replicates=replicates, seed=0)
    seconds = time.perf_counter() - began
    rate = seconds / (rows * replicates)
    measures.add(
        "bootstrap",
        "seconds_per_row_replicate",
        f"rows={rows},replicates={replicates}",
        rate,
        "s",
    )
    return rate


def measure_installs(executor: ConnectionExecutor, measures: Measures) -> None:
    """The seconds to install each query again with INSTALL QUERY -FORCE, one at a time, which
    leaves each query as it was: the same text, installed anew. A query that does not install
    is named, and the rest go on (mule install installs what is not current)."""
    for name in QUERY_FILES:
        began = time.perf_counter()
        note = ""
        try:
            with executor.client.request_timeout(read_s=INSTALL_SECONDS):
                executor.call(
                    lambda conn, name=name: conn.installQueries([name], flag="-force"),
                    what="installQueries",
                    attempts=1,
                )
        except Exception as error:
            note = f"{type(error).__name__}: {str(error)[:200]}"
        measures.add(
            "install",
            "seconds",
            name,
            time.perf_counter() - began if not note else float("nan"),
            "s",
            note,
        )


def measure_server(executor: ConnectionExecutor, world: World, measures: Measures) -> None:
    """TigerGraph's version and the loaded world's counts of vertices and edges. Its memory in
    use is not measured here: the metrics endpoint answers 403 for this workspace's secret, so
    it is read in the Cloud console."""
    version = str(executor.call(lambda conn: conn.getVer(), what="getVer"))
    measures.add("world", "tigergraph", "", 1.0, "version", version)
    for name, count in world.vertices.items():
        measures.add("world", "vertices", name, count, "vertices")
    for name, count in world.edges.items():
        measures.add("world", "edges", name, count, "edges")


def measure_co_users(
    samples: Sequence[Sample],
    cache: Cache,
    measures: Measures,
    keys: dict[str, list[tuple[int, str]]],
) -> None:
    """What the choice of co-user rows per Monday for every customer, against on demand per
    scored account, turns on: the co-user accounts a subgraph holds, how many of those are in
    more than one sampled subgraph of a Monday (the reuse a per-Monday computation would
    share), and the customers a Monday has. The rule is the same either way."""
    held = [key for group in keys.values() for key in group]
    if not held:
        return
    rows = read_rows(cache, held)
    per_subgraph = [len(example.identity.accounts) for example in rows]
    measures.spread("co_users", "co_user_accounts", "sample", per_subgraph, "accounts")
    by_monday: dict[int, list[set[str]]] = {}
    for (t_ms, _), example in zip(held, rows, strict=True):
        by_monday.setdefault(t_ms, []).append({a.account_id for a in example.identity.accounts})
    total = shared = 0
    for sets in by_monday.values():
        seen: dict[str, int] = {}
        for accounts in sets:
            for account in accounts:
                seen[account] = seen.get(account, 0) + 1
        total += len(seen)
        shared += sum(1 for count in seen.values() if count > 1)
    measures.add(
        "co_users",
        "shared_between_subgraphs",
        "sample",
        shared / total if total else 0.0,
        "share",
        "of the distinct co-user accounts of a Monday's sampled subgraphs",
    )
    for sample in samples:
        measures.add(
            "co_users", "scorable_accounts", f"t={sample.t_ms}", sample.scorable, "accounts"
        )


# The projections.


def project(
    config: RunConfig,
    loaded: LoadedWorld,
    samples: Sequence[Sample],
    rates: Rates,
    fill_rate: float,
    handoff_rate: float,
    bootstrap_rate: float,
    measures: Measures,
) -> None:
    """The projections of the module's docstring, from what was measured."""
    mondays = len(config.calendar.mondays())
    scorable = float(np.mean([sample.scorable for sample in samples]))
    per_person = scorable / loaded.config.people
    identified = float(np.mean([sample.identified_active for sample in samples]))
    identified_per_person = identified / loaded.config.people
    grid = len(config.training.prior_multiples)
    runs = runs_of(
        len(config.training.seeds), len(ablations(config.training)), len(XGBOOST_RANKERS)
    )
    measures.add("selection", "scorable_accounts_per_person_monday", "", per_person, "accounts")
    measures.add("selection", "build_seconds_per_example", "", rates.build, "s")
    measures.add("selection", "forward_seconds_per_example", "", rates.forward, "s")
    passes = {
        "grid": selection_passes(config.training, grid),
        "fixed": selection_passes(config.training, 1),
    }
    reuse = config.training.reuse_built_examples
    for name, count in passes.items():
        measures.add("selection", "passes_per_rotation", name, count, "passes")
    measures.add(
        "bootstrap", "runs", "", runs, "runs", "seeds x (full model + ablations) + XGBoost"
    )
    measures.add("bootstrap", "replicates", "", config.audit.replicates, "replicates")
    for people in sorted({loaded.config.people, *config.bench.projected_people}):
        variant = f"people={people}"
        examples = examples_at(per_person, people, mondays)
        measures.add("selection", "examples_per_pass", variant, examples, "examples")
        for name, count in passes.items():
            built, scored = pass_seconds(rates.build, rates.forward, examples, count, reuse=reuse)
            for cost, seconds in (("build", built), ("build+forward", scored)):
                label = f"{variant},priors={name},cost={cost}"
                measures.add(
                    "selection",
                    "projected_seconds_per_rotation",
                    label,
                    seconds,
                    "s",
                    "built once and kept" if reuse else "built anew on every pass",
                )
                measures.add(
                    "selection",
                    "projected_days_per_rotation",
                    label,
                    seconds / SECONDS_PER_DAY,
                    "days",
                )
        sparse = config.audit.nonmule_sample + examples_at(identified_per_person, people, mondays)
        measures.add(
            "audit",
            "account_mondays_fetched_at_least",
            variant,
            sparse,
            "account-Mondays",
            "the sample plus the identified active mules, a lower bound on the true mules'",
        )
        measures.add(
            "audit",
            "projected_fetch_seconds_at_least",
            variant,
            seconds_projected(fill_rate, sparse),
            "s",
        )
        measures.add(
            "bootstrap",
            "projected_seconds",
            variant,
            config.audit.replicates * runs * examples * bootstrap_rate,
            "s",
            f"{config.audit.replicates} replicates x {runs} runs x {examples:,.0f} ranked rows",
        )
        mules = examples_at(identified_per_person, people, mondays)
        fallback = examples / config.data.fallback_weight + mules
        for name, count in (("whole", examples), ("fallback", fallback)):
            label = f"{variant},scoring={name}"
            measures.add("scoring", "account_mondays", label, count, "account-Mondays")
            measures.add(
                "scoring", "fetch_seconds_at_most", label, seconds_projected(fill_rate, count), "s"
            )
            measures.add(
                "scoring",
                "scoring_pass_seconds",
                label,
                seconds_projected(rates.scoring, count),
                "s",
                "one run's pass over the test keys",
            )
            measures.add(
                "scoring",
                "test_scoring_seconds_all_runs",
                label,
                pass_seconds(rates.build, rates.forward, count, runs, reuse=reuse)[1],
                "s",
                f"{runs} runs, "
                + ("the test batches built once and kept" if reuse else "each building them anew"),
            )
        measures.add(
            "handoff",
            "projected_matrix_bytes",
            variant,
            examples * len(GRAPH_COLUMNS) * 4,
            "bytes",
            "one selection or test matrix of the graph features, float32",
        )
        measures.add(
            "handoff",
            "projected_score_seconds_per_matrix",
            variant,
            seconds_projected(handoff_rate, examples),
            "s",
        )


# The command.


def run(
    executor: ConnectionExecutor,
    config: RunConfig = DEFAULT_CONFIG,
    *,
    records: Sequence[Path] | None = None,
    folder: Path | None = None,
    installs: bool = True,
) -> tuple[LoadedWorld, pd.DataFrame, Path]:
    """Run mule bench on the loaded world: measure it, project it, and write the table to
    folder (results/bench by default) as <world name>.parquet. records are the identity files
    to find the world among, as in mule prepare; installs says whether to time a forced
    reinstall of every query. The loaded world, the table and its path."""
    bench = BenchExecutor(executor)
    began = time.perf_counter()
    loaded = find_world(bench, config.worlds, records)
    world_seconds = time.perf_counter() - began
    check_window(loaded.world, config.calendar)
    measures = Measures()
    measures.add("world", "read_world_seconds", "", world_seconds, "s")
    measures.add("world", "people", loaded.config.name, loaded.config.people, "people")
    measure_server(executor, loaded.world, measures)
    labels = read_identified(bench)
    measures.add("world", "identified_mules", "", len(labels.mules), "accounts")
    mondays = chosen_mondays(config.calendar.mondays(), config.bench.mondays)
    samples = take_samples(bench, mondays, labels, config, measures)
    show(f"bench: {len(samples)} Mondays, measuring {loaded.config.name}")
    measure_fetch(bench, samples, config, measures)
    with tempfile.TemporaryDirectory(prefix="mule-bench-cache-") as name:
        cache = Cache(Path(name) / "cache")
        keys = fill_cache(bench, cache, samples, config, measures)
        measure_subgraphs(cache, keys, measures)
        measure_co_users(samples, cache, measures, keys)
        rates, batches = measure_model(cache, identity(loaded.world), keys, config, measures)
        handoff = measure_handoff(batches, config, measures)
    measure_profile(bench, samples, config, measures)
    bootstrap = measure_bootstrap(config, measures)
    if installs:
        measure_installs(executor, measures)
    # The sparse fetch rate over both groups: all the seconds over all the accounts.
    spent = sum(row[3] for row in measures.rows if row[:2] == ("fetch", "fill_seconds"))
    filled = sum(row[3] for row in measures.rows if row[:2] == ("fetch", "fill_accounts"))
    project(config, loaded, samples, rates, spent / filled, handoff, bootstrap, measures)
    table = measures.frame()
    target = (bench_dir() if folder is None else folder) / f"{loaded.config.name}.parquet"
    write_parquet(target, pa.Table.from_pandas(table, preserve_index=False))
    return loaded, table, target


def lines(table: pd.DataFrame) -> list[str]:
    """The headline numbers of a bench table as a person reads them: what selection, the audit
    and the bootstrap cost at each size."""
    found: list[str] = []
    for section, name, label in (
        ("selection", "projected_days_per_rotation", "selection, days per rotation"),
        ("audit", "projected_fetch_seconds_at_least", "audit fetch, seconds at least"),
        ("bootstrap", "projected_seconds", "bootstrap, seconds"),
    ):
        held = table[(table["section"] == section) & (table["name"] == name)]
        found += [f"{label}, {row.variant}: {row.value:,.1f}" for row in held.itertuples()]
    return found
