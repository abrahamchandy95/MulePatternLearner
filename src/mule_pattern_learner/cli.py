"""mule: find money-mule accounts with a temporal graph neural network on TigerGraph (docs/design.md,
"Commands" and "The pipeline").

Nine commands, none with an option: worlds, roles, dates and every setting live in config.py
(DEFAULT_CONFIG), and any argument after a command is refused.

    mule install    installs the queries that are not current
    mule parity     compares every query with the reference graph, writes gsql/parity.toml
    mule prepare    matches the loaded world, checks its calendar and labels, writes the examples
    mule fetch      fills the cache for every example the loaded world's roles need
    mule audit      reads the answer key, runs the label checks, clock gain test and feature audit
    mule train      trains the GNN, its ablations and the two XGBoost comparisons
    mule evaluate   scores the test world, runs the clock invariance test, and writes the
                    metrics and the missed-mule report
    mule score      scores the configured accounts and dates with the best-selection model
    mule bench      measures what the loaded world costs, and projects it

The commands that read the loaded world (parity, prepare, fetch, audit, bench and score) first run
a readiness check that changes nothing in the graph or in the cache (check_ready): the installed
queries are the repository's, the loaded world is a configured one, one account is fetched into
a scratch folder, and one training step runs. mule train and mule evaluate never query
TigerGraph, so theirs (check_trainable, check_evaluable) reads what they read: the worlds of every
rotation are registered, one example is read from the cache of each world the command reads
(mule train the training and the selection world, mule evaluate the test world), and one training
step runs. Both run for every rotation before the first rotation starts, so a cache that is not
there is found before any rotation trains or is evaluated.

A rotation is three configured worlds, one that trains, one that selects and one that tests, and
the results of one are in results/<train>-<select>-<test>/ (run_name). mule train and mule
evaluate run every rotation of config.worlds.rotations in order, so a config that holds one
rotation runs one. mule train first reads the pass or fail of the audit records of every
rotation's worlds and stops, naming the checks, when one stops the run (evaluation.audit.
train_may_run, the one reading of audit records the design allows before training); no training
code reads one. The worlds are found by their configured names in the registry mule prepare
writes (data.worlds.registered), since only one world is loaded in TigerGraph at a time.

Each command appends to an events.jsonl (started, finished, or stopped with the error): in the
run's results folder for train and evaluate, in results/bench/ for bench, and in results/commands/
for the rest.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import sys
import tempfile
import time

import pandas as pd
import pyarrow as pa
import torch
from torch.optim import Adam

from . import bench
from .comparisons import rankers
from .comparisons.features import vectors_of
from .config import DEFAULT_CONFIG, Rotation, RunConfig, WorldConfig
from .data.calendar import check_window, epoch_ms
from .data.examples import (
    SELECTION_FILE,
    SELECTION_SCHEMA,
    TEST_FILE,
    TEST_SCHEMA,
    TRAINING_FILE,
    TRAINING_SCHEMA,
    ExampleKeys,
    examples_folder,
    examples_key,
    read_first,
    read_table,
)
from .data.labels import read_identified
from .data.prepare import prepare
from .data.report import REPORT_FILE, REPORT_SCHEMA, identified_active_share, lines
from .data.worlds import (
    SELECT,
    TEST,
    TRAIN,
    LoadedWorld,
    Registered,
    find_world,
    register,
    registered,
    roles,
)
from .evaluation import audit as auditing
from .evaluation.audit import DEVELOPMENT, audit_settings, train_may_run
from .evaluation.bias import rescore_epochs
from .evaluation.clock_invariance import Invariance, invariance
from .evaluation.rankings import score_gnn, score_trees
from .evaluation.report import (
    GNN,
    MadeUnder,
    Run,
    ScoredWorld,
    Verdict,
    append_event,
    evaluate,
    manifest,
    rotation_spread,
    selected,
    verdict,
    write_manifest,
)
from .evaluation.store import Store, write_parquet
from .model.encode import encode_batch
from .model.hetero import HeteroModel
from .model.loss import NonNegativePULoss
from .paths import bench_dir, results_dir
from .runtime.console import duration, plural, show
from .subgraph.build import Batch, build, read_rows
from .subgraph.cache import Cache
from .subgraph.fetch import Fetcher, fetch
from .subgraph.layouts import LAYOUT
from .subgraph.switches import Switch
from .testing import parity
from .tigergraph import installer
from .tigergraph.connection import ConnectionSettings
from .tigergraph.executor import ConnectionExecutor, FetchExecutor, TigerGraphExecutor
from .tigergraph.queries import LIST_ACCOUNTS, VERSIONS, list_accounts
from .tigergraph.world import cache_key
from .training.batches import ScoringBatches, training_batches
from .training.priors import Decision, decide, grid
from .training.scoring import best_seed, score
from .training.selection import ranked_keys
from .training.trainer import (
    EPOCHS,
    SelectionWorld,
    Trained,
    ablations,
    gnn_lock,
    load_trained,
    model_file,
    ranker_name,
    train_rotation,
)

# The commands of the design ("Commands"), in the order a run uses them, with what each does. None
# takes an option: worlds, roles, dates and settings live in config.
COMMANDS = {
    "install": "install the changed queries on TigerGraph",
    "parity": "compare every query with the reference graph and write the parity record",
    "prepare": "match the loaded world, check its calendar and labels, write the examples",
    "fetch": "fill the cache for every example the loaded world's roles need",
    "audit": "read the answer key and run the label checks, clock gain test and feature audit",
    "train": "train the GNN, its ablations and the two XGBoost comparisons",
    "evaluate": "score the test world, test the clock invariance, and write the metrics and "
    "the missed-mule report",
    "score": "score the configured accounts and dates with the best-selection model",
    "bench": "measure the cost of fetching, building and training on the loaded world",
}
# The folder, under results/, of the events of the commands that have no run of their own.
COMMANDS_RUN = "commands"
# The folder under results/ of what is measured over rotations.
ROTATIONS_RUN = "rotations"
# The seed whose every epoch's model a development rotation keeps (docs/design.md, "Training and
# selection"), and the file of the decision on the prior grid.
EPOCH_SEED = 0
DECISION_FILE = "prior_decision.txt"
SCORES_FILE = "scores.parquet"
# The world of mule score's examples, which match no world: a name for their keys alone, which
# never reaches the model.
SCORED_WORLD = "scored"
# The examples tables each role reads.
ROLE_TABLES = {TRAIN: TRAINING_FILE, SELECT: SELECTION_FILE}
# What the tuned settings of a manifest were chosen from (docs/design.md, "Results").
CHOSEN_FROM = {
    "epoch": "the selection score on the selection world, from pu_label alone",
    "prior": "the selection score on the development rotation's grid, from pu_label alone",
    "xgboost_rounds": "the selection score on the selection world, from pu_label alone",
}


class CommandError(RuntimeError):
    """A command that stopped for a reason a person can act on: the message says why."""


# What ends a command with its message and status 1 instead of a traceback: its own stops, a
# refusal (ValueError, which the checks of this package raise), a missing file, and a failure of
# the transport, of an XGBoost process or of an install (RuntimeError, TimeoutError), each of
# which says why. Anything else is a bug and keeps its traceback.
STOPS = (CommandError, ValueError, FileNotFoundError, RuntimeError, TimeoutError)


class NotReady(CommandError):
    """The readiness check failed: the message names the check and its cause."""


type Connect = Callable[[], ConnectionExecutor]


def connect() -> ConnectionExecutor:
    """The executor of the TigerGraph workspace the repository's .env names."""
    return TigerGraphExecutor(settings=ConnectionSettings())


def run_name(rotation: Rotation) -> str:
    """The name of a rotation's results folder: its three worlds, joined by dashes."""
    return f"{rotation.train}-{rotation.select}-{rotation.test}"


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# The readiness check.


def _step(batch: Batch, config: RunConfig, device: torch.device) -> None:
    """One training step of a fresh model on a batch with its label and weight, which changes
    nothing outside the process: the model, the loss and the device work."""
    torch.manual_seed(0)
    model = HeteroModel(LAYOUT, config.model).to(device)
    optimiser = Adam(model.parameters(), lr=config.training.learning_rate)
    data = encode_batch(batch.data, LAYOUT, device)
    loss, _ = NonNegativePULoss(0.01)(model(data), data.label, data.weight)
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
    if not torch.isfinite(loss):
        raise NotReady(f"one training step: the loss is {loss.item()}, not finite")


def check_queries(executor: ConnectionExecutor) -> None:
    """The first check of every command that reads the graph, and the only one of mule score,
    which matches no world: the installed queries are the repository's (installer.verify)."""
    try:
        installer.verify(executor)
    except ValueError as error:
        raise NotReady(f"installed queries: {error}") from error


def check_ready(executor: ConnectionExecutor, config: RunConfig) -> LoadedWorld:
    """The read-only readiness check of a command that reads the loaded world: the installed
    queries are the repository's (installer.verify), the loaded world is a configured one that
    holds the calendar's window, one account is fetched into a scratch folder and built into an
    example, and one training step runs on it. Each failure names its check. The loaded world."""
    check_queries(executor)
    try:
        loaded = find_world(FetchExecutor(executor), config.worlds)
        check_window(loaded.world, config.calendar)
    except (ValueError, FileNotFoundError) as error:
        raise NotReady(f"the world's identity: {error}") from error
    show(f"ready: the loaded world is {loaded.config.name} ({loaded.identity})")
    fetching = FetchExecutor(executor)
    for t_ms in config.calendar.scoring_times(loaded.config.seed):
        found = list_accounts.parse(
            fetching.run(LIST_ACCOUNTS, list_accounts.params(t_ms, "", 1)), "", 1
        )
        if found:
            break
    else:
        raise NotReady("one account fetched: the loaded world has no scorable account")
    account = found[0].account_id
    with tempfile.TemporaryDirectory(prefix="mule-ready-") as folder:
        cache = Cache(Path(folder))
        try:
            Fetcher(executor, cache, config.transport).week(t_ms, [account])
            rows = read_rows(cache, [(t_ms, account)])
        except (ValueError, KeyError, RuntimeError) as error:
            raise NotReady(f"one account fetched: {error}") from error
    batch = build(rows, loaded.identity, labels=[1], weights=[1.0])
    show(f"ready: fetched {account} at {t_ms}")
    _step(batch, config, _device())
    show("ready: one training step ran")
    return loaded


@dataclass(frozen=True)
class RotationWorlds:
    """The three worlds of a rotation as mule prepare registered them, in the order of their
    roles: each with its configured settings and its identity and label identity."""

    rotation: Rotation
    train: tuple[WorldConfig, Registered]
    select: tuple[WorldConfig, Registered]
    test: tuple[WorldConfig, Registered]

    @property
    def worlds(self) -> list[tuple[WorldConfig, Registered]]:
        return [self.train, self.select, self.test]

    @property
    def name(self) -> str:
        return run_name(self.rotation)


def rotation_worlds(rotation: Rotation, config: RunConfig) -> RotationWorlds:
    """The worlds of a rotation, found by name in the registry; a world that was never
    prepared is refused."""
    configured = {world.name: world for world in config.worlds.worlds}
    try:
        found = [
            (configured[name], registered(name))
            for name in (rotation.train, rotation.select, rotation.test)
        ]
    except ValueError as error:
        raise NotReady(f"the world's identity: {error}") from error
    return RotationWorlds(rotation, *found)


def _first_example(
    world: WorldConfig, held: Registered, config: RunConfig, keys: Callable[[], pd.DataFrame]
) -> Batch:
    """The readiness check of the commands that never query TigerGraph: the first example of a
    world's keys is read from its cache and built, which is what the command's first pass over
    the world reads. The batch."""
    try:
        first = keys().iloc[0]
        cache = Cache.of(held.identity, config.subgraph)
        rows = read_rows(cache, [(int(first["t_ms"]), str(first["account_id"]))])
    except (FileNotFoundError, KeyError, IndexError, OSError) as error:
        raise NotReady(
            f"one example read: the cache and examples of {world.name} are not there: {error}"
        ) from error
    show(f"ready: read the example of {first['account_id']} of {world.name}")
    return build(rows, held.identity, labels=[1], weights=[1.0])


def check_trainable(worlds: RotationWorlds, config: RunConfig) -> None:
    """The readiness check of mule train on a rotation: the first example of the training world
    and the first ranked example of the selection world are read from their caches (selection
    reads its examples lazily, so a selection world that was never fetched would otherwise fail
    after the first epoch), and one training step runs on the training one."""
    key = examples_key(config.calendar, config.data)
    train_world, train_held = worlds.train
    train_folder = examples_folder(train_held.identity, key, train_held.labels)
    batch = _first_example(
        train_world,
        train_held,
        config,
        lambda: read_first(train_folder / TRAINING_FILE, TRAINING_SCHEMA),
    )
    select_world, select_held = worlds.select
    select_folder = examples_folder(select_held.identity, key, select_held.labels)
    _first_example(
        select_world,
        select_held,
        config,
        lambda: ranked_keys(read_first(select_folder / SELECTION_FILE, SELECTION_SCHEMA)),
    )
    _step(batch, config, _device())
    show("ready: one training step ran")


def check_evaluable(worlds: RotationWorlds, config: RunConfig) -> None:
    """The readiness check of mule evaluate on a rotation: the test world's first example is
    read from its cache, and one training step runs on it."""
    world, held = worlds.test
    batch = _first_example(world, held, config, lambda: _test_keys(worlds, config, read_first))
    _step(batch, config, _device())
    show("ready: one training step ran")


# The commands.


def install_command(config: RunConfig, open_graph: Connect) -> None:
    executor = open_graph()
    began = time.monotonic()
    done = installer.install(executor)
    if done:
        show(
            f"installed {plural(len(done), 'query', 'queries')} in "
            f"{duration(time.monotonic() - began)}: {', '.join(done)}"
        )
    else:
        show("every query is current: nothing to install")


def parity_command(config: RunConfig, open_graph: Connect) -> None:
    executor = open_graph()
    loaded = check_ready(executor, config)
    found = parity.run(executor, config, seed=loaded.config.seed)
    for line in found.lines():
        show(line)
    if found.failed:
        raise CommandError(
            f"{plural(len(found.failed), 'query', 'queries')} did not match the reference graph: "
            f"{', '.join(found.failed)}; the parity record holds the {len(found.passed)} that did"
        )
    show(f"every query matched the reference graph; the record holds {len(found.passed)} versions")


def prepare_command(config: RunConfig, open_graph: Connect) -> None:
    executor = open_graph()
    check_ready(executor, config)
    prepared = prepare(executor, config)
    register(prepared.world, prepared.labels.identity)
    show(
        f"prepared {prepared.world.config.name} ({prepared.world.identity}), roles "
        f"{', '.join(sorted(prepared.roles)) or 'none'}, in {prepared.folder}"
    )
    for line in lines(prepared.report):
        show(line)


def role_tables(played: frozenset[str], scoring: str) -> list[str]:
    """The examples tables whose keys mule fetch fetches: the training and selection tables of
    the roles the world plays, and the test table when it tests under whole scoring (the
    fallback list is mule audit's)."""
    tables = [ROLE_TABLES[role] for role in (TRAIN, SELECT) if role in played]
    if TEST in played and scoring == "whole":
        tables.append(TEST_FILE)
    return tables


def fetch_command(config: RunConfig, open_graph: Connect) -> None:
    executor = open_graph()
    loaded = check_ready(executor, config)
    labels = read_identified(FetchExecutor(executor))
    folder = examples_folder(
        loaded.identity, examples_key(config.calendar, config.data), labels.identity
    )
    played = roles(loaded.config.name, config.worlds.rotations)
    tables = role_tables(played, config.data.scoring)
    if not tables:
        show(
            f"nothing to fetch: {loaded.config.name} plays no role that needs its examples in any "
            "rotation of config.worlds.rotations"
        )
        return
    try:
        keys = ExampleKeys(folder, tables)
    except FileNotFoundError as error:
        raise CommandError(str(error)) from error
    began = time.monotonic()
    done = fetch(executor, Cache.of(loaded.identity, config.subgraph), keys, config.transport)
    show(
        f"fetched {plural(len(keys), 'week')} of {loaded.config.name} in "
        f"{duration(time.monotonic() - began)}: "
        + ", ".join(f"{name} {done.accounts[name]:,}" for name in sorted(done.accounts))
    )


def audit_command(config: RunConfig, open_graph: Connect) -> None:
    executor = open_graph()
    check_ready(executor, config)
    found = auditing.audit(executor, config)
    record = found.record
    show(f"audited {record.world} ({record.identity}): {len(record.checks)} checks")
    for name, passed in sorted(record.checks.items()):
        show(f"  {'passed' if passed else 'FAILED'}  {name}")
    for name, reasons in found.reasons.items():
        for reason in reasons[:5]:
            show(f"  {name}: {reason}")
    if record.failed:
        raise CommandError(
            f"{plural(len(record.failed), 'check')} failed: {', '.join(record.failed)}; the pass "
            "record was written, and mule train stops on the ones that stop it"
        )


def _decision(trained: Mapping[str, list[Trained]], config: RunConfig) -> Decision | None:
    """The prior grid's decision from the full model's selected epochs, when the grid holds
    more than one prior."""
    full = trained[GNN]
    if len(full[0].runs) < 2:
        return None
    found = {
        run.prior.multiple: [
            next(held.found for held in model.runs if held.prior.multiple == run.prior.multiple)
            for model in full
        ]
        for run in full[0].runs
    }
    return decide(found, config.audit, config.training)


def _write_text(path: Path, text: str) -> None:
    """A file written through a temporary file and a rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(text)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def train_rotation_command(worlds: RotationWorlds, config: RunConfig, device: torch.device) -> None:
    """Train one rotation: the full model over the prior grid and every seed, each ablation at
    the prior the full model chose, and the two XGBoost comparisons once, each selected on the
    selection world; the model files go to the rotation's results folder. With the proof of
    concept's settings (training.ablations off, training.fixed_prior set) there is no ablation and
    the grid is the one prior."""
    train_world, train_registered = worlds.train
    _, select_registered = worlds.select
    key = examples_key(config.calendar, config.data)
    train_folder = examples_folder(train_registered.identity, key, train_registered.labels)
    select_folder = examples_folder(select_registered.identity, key, select_registered.labels)
    examples = read_table(train_folder / TRAINING_FILE, TRAINING_SCHEMA)
    report = read_table(train_folder / REPORT_FILE, REPORT_SCHEMA)
    selection_keys = read_table(select_folder / SELECTION_FILE, SELECTION_SCHEMA)
    batch_size = config.training.batch_size
    show(f"{worlds.name}: building the training batches of {train_world.name}")
    batches = training_batches(
        Cache.of(train_registered.identity, config.subgraph),
        train_registered.identity,
        examples,
        batch_size,
        seed=config.training.shuffle_seed,
    )
    selection = SelectionWorld(
        Cache.of(select_registered.identity, config.subgraph),
        select_registered.identity,
        selection_keys,
        batch_size,
        keep=config.training.reuse_built_examples,
    )
    priors = grid(identified_active_share(report), config.training)
    folder = results_dir(worlds.name)
    development = train_world.kind == DEVELOPMENT
    show(f"{worlds.name}: training the GNN and its ablations over {len(priors)} priors")
    trained = train_rotation(
        batches,
        selection,
        priors,
        folder,
        model=config.model,
        settings=config.training,
        subgraph=config.subgraph,
        device=device,
        keep_epochs_of_seed=EPOCH_SEED
        if development and EPOCH_SEED in config.training.seeds
        else None,
    )
    decision = _decision(trained, config) if development else None
    if decision is not None:
        _write_text(folder / DECISION_FILE, "\n".join(decision.lines()) + "\n")
        for line in decision.lines():
            show(f"{worlds.name}: {line}")
    show(f"{worlds.name}: training the XGBoost comparisons")
    training_vectors = vectors_of(batches)
    selection_vectors = vectors_of(selection.batches)
    for name in rankers.COLUMNS:
        trees = rankers.train(name, training_vectors, selection_vectors, selection.keys)
        rankers.save(trees, rankers.model_file(folder, name), rankers.lock(name, config.subgraph))
        show(f"{worlds.name}: {name} kept {trees.rounds} rounds, selection score {trees.score:.4f}")


def _logged[T](folder: Path, command: str, work: Callable[[], T]) -> T:
    """Run work, appending to the events of folder that it started and how it ended; what it
    returns."""
    append_event(folder, command, "started")
    try:
        found = work()
    except BaseException as error:
        append_event(folder, command, "stopped", error)
        raise
    append_event(folder, command, "finished")
    return found


def _gates(world: WorldConfig, config: RunConfig) -> bool:
    """Whether a world's failed or missing audit stops mule train: a reported world's always, a
    development world's unless training.development_audit_stops is off (proof of concept)."""
    return world.kind != DEVELOPMENT or config.training.development_audit_stops


def train_command(config: RunConfig, open_graph: Connect) -> None:
    del open_graph  # training never queries TigerGraph
    rotations = [rotation_worlds(rotation, config) for rotation in config.worlds.rotations]
    # Each world once, though a world is in up to three rotations.
    audited = {
        world.name: (world, held.identity)
        for rotation in rotations
        for world, held in rotation.worlds
    }
    stops = train_may_run([held for held in audited.values() if _gates(held[0], config)])
    if stops:
        raise CommandError(
            f"training stops: {'; '.join(stops)}. Run `mule audit` on each world, and fix a check "
            "that failed"
        )
    for worlds in rotations:
        check_trainable(worlds, config)
    device = _device()
    for worlds in rotations:
        folder = results_dir(worlds.name)
        ungated = [
            (world, held.identity) for world, held in worlds.worlds if not _gates(world, config)
        ]
        for stop in train_may_run(ungated):
            show(f"{worlds.name}: audit not passed, and training goes on: {stop}")
            append_event(folder, "train", f"audit not passed: {stop}")
        began = time.monotonic()
        _logged(
            folder, "train", lambda worlds=worlds: train_rotation_command(worlds, config, device)
        )
        show(f"{worlds.name}: trained in {duration(time.monotonic() - began)}")


def _runs(
    worlds: RotationWorlds, config: RunConfig, batches: ScoringBatches, device: torch.device
) -> list[Run]:
    """Every run of a rotation that mule train wrote: each seed of the full model and of each
    ablation, and each XGBoost comparison, with its selection score and its scores of the test
    keys as a function that scores them when evaluate asks. A model file mule train did not
    write is refused."""
    folder = results_dir(worlds.name)
    runs: list[Run] = []
    expected: list[tuple[Switch | None, int]] = [
        (switch, seed)
        for switch in (None, *ablations(config.training))
        for seed in config.training.seeds
    ]
    for switch, seed in expected:
        path = model_file(folder, ranker_name(switch), seed)
        if not path.is_file():
            raise CommandError(f"{path} is missing: run `mule train`")
        lock = gnn_lock(switch, config.model, config.subgraph, config.training)
        saved = load_trained(path, lock, device)
        runs.append(
            Run(
                ranker_name(switch),
                seed,
                saved.score,
                lambda path=path, lock=lock, switch=switch: score_gnn(
                    path, lock, batches, switch, device
                ),
            )
        )
    for name in rankers.COLUMNS:
        path = rankers.model_file(folder, name)
        if not path.is_file():
            raise CommandError(f"{path} is missing: run `mule train`")
        lock = rankers.lock(name, config.subgraph)
        runs.append(
            Run(
                name,
                None,
                rankers.load(path, lock).score,
                lambda path=path, lock=lock: score_trees(path, lock, batches),
            )
        )
    return runs


def _test_keys(
    worlds: RotationWorlds,
    config: RunConfig,
    read: Callable[[Path, pa.Schema], pd.DataFrame] = read_table,
) -> pd.DataFrame:
    """The test world's keys with their weights (t_ms, account_id, weight): every scorable
    account of every week under whole scoring, each weighing 1, or the fallback list mule audit
    stored, with its weights. read reads the examples table, read_table (the whole table) unless
    the readiness check asks only for its first rows (read_first)."""
    _, test_registered = worlds.test
    if config.data.scoring == "fallback":
        return Store(test_registered.identity).fallback()
    folder = examples_folder(
        test_registered.identity, examples_key(config.calendar, config.data), test_registered.labels
    )
    return read(folder / TEST_FILE, TEST_SCHEMA).assign(weight=1.0)


def _invariance(
    worlds: RotationWorlds,
    config: RunConfig,
    scored: ScoredWorld,
    runs: Sequence[Run],
    device: torch.device,
) -> Invariance | None:
    """The clock invariance test (evaluation.clock_invariance) of the GNN mule score uses, the
    seed with the best selection score, on the audit sample mule audit stored for the test
    world; None when mule audit stored none."""
    stored = Store(scored.identity).clock_sample()
    if stored is None:
        return None
    best = selected(runs)
    if best.seed is None:
        raise ValueError("The selected GNN has no seed")
    path = model_file(results_dir(worlds.name), ranker_name(None), best.seed)
    lock = gnn_lock(None, config.model, config.subgraph, config.training)

    def score_keys(keys: pd.DataFrame) -> pd.DataFrame:
        batches = ScoringBatches(scored.cache, scored.identity, keys, config.training.batch_size)
        return score_gnn(path, lock, batches, None, device)

    return invariance(
        stored,
        scored.key,
        scored.scoring_times,
        config.calendar.week_starts(),
        score_keys,
        config.audit,
        config.training.bootstrap_seed,
    )


def _verdict(worlds: RotationWorlds, config: RunConfig, tested: Invariance | None) -> Verdict:
    """Whether the rotation's result may be reported (evaluation.report.verdict): the worlds of
    the rotation, the development worlds that mule prepare registered, and the clock invariance
    test of the rotation's test world."""
    development: list[tuple[WorldConfig, str]] = []
    kinds = {world.name: world for world in config.worlds.worlds}
    for rotation in config.worlds.rotations:
        names = (rotation.train, rotation.select, rotation.test)
        if all(kinds[name].kind == DEVELOPMENT for name in names):
            for name in names:
                try:
                    development.append((kinds[name], registered(name).identity))
                except ValueError:
                    continue
    current = MadeUnder(dict(VERSIONS), cache_key(config.subgraph), audit_settings(config))
    return verdict(
        [(world, held.identity) for world, held in worlds.worlds], development, current, tested
    )


def evaluate_rotation(
    worlds: RotationWorlds, config: RunConfig, device: torch.device
) -> pd.DataFrame:
    """Evaluate one rotation: score the test world with every model file of its results folder,
    one run at a time, run the clock invariance test of the selected GNN on the sample mule audit
    stored, write the tables, the missed-mule report and the manifest. The metrics table."""
    began = time.monotonic()
    test_world, test_registered = worlds.test
    folder = results_dir(worlds.name)
    keys = _test_keys(worlds, config)
    cache = Cache.of(test_registered.identity, config.subgraph)
    batches = ScoringBatches(
        cache,
        test_registered.identity,
        keys.loc[:, ["t_ms", "account_id"]],
        config.training.batch_size,
        keep=config.training.reuse_built_examples,
    )
    scored = ScoredWorld.of(
        test_registered.identity, cache, keys, config.calendar.scoring_times(test_world.seed)
    )
    runs = _runs(worlds, config, batches, device)
    epochs = None
    if worlds.train[0].kind == DEVELOPMENT and (folder / EPOCHS).is_dir():
        select_world, select_registered = worlds.select
        selection_table = read_table(
            examples_folder(
                select_registered.identity,
                examples_key(config.calendar, config.data),
                select_registered.labels,
            )
            / SELECTION_FILE,
            SELECTION_SCHEMA,
        )
        ranked = ranked_keys(selection_table)
        chosen = Store(select_registered.identity)
        epochs = rescore_epochs(
            sorted((folder / EPOCHS).glob("*.pt")),
            gnn_lock(None, config.model, config.subgraph, config.training),
            ScoringBatches(
                Cache.of(select_registered.identity, config.subgraph),
                select_registered.identity,
                ranked,
                config.training.batch_size,
                keep=config.training.reuse_built_examples,
            ),
            ranked.loc[:, ["t_ms", "account_id", "weight"]],
            chosen.answer_key(),
            chosen.labels().accounts,
            config.calendar.scoring_times(select_world.seed),
            device,
        )
    tested = _invariance(worlds, config, scored, runs, device)
    found = _verdict(worlds, config, tested)
    append_event(
        folder,
        "evaluate",
        "clock invariance not run"
        if tested is None
        else f"clock invariance {'passed' if tested.passed else 'not passed'}",
    )
    tables = evaluate(
        scored,
        runs,
        config.audit,
        config.training.bootstrap_seed,
        folder,
        rotation=worlds.name,
        verdict_found=found,
        epochs=epochs,
        invariance=tested,
    )
    write_manifest(
        folder,
        manifest(
            verdict_found=found,
            identities={world.name: held.identity for world, held in worlds.worlds},
            wall_seconds={"evaluate": time.monotonic() - began},
            tigergraph_calls=0,
            cache_hits=0,
            chosen_from=CHOSEN_FROM,
            informed_by=[],
        ),
    )
    show(
        f"{worlds.name}: evaluated {test_world.name} with {plural(len(runs), 'run')}, "
        + (
            "a reported result"
            if found.reported
            else "development only: " + "; ".join(found.reasons[:3])
        )
    )
    return tables["metrics"]


def rotation_spreads(
    metrics: Mapping[str, pd.DataFrame], kinds: Mapping[str, str]
) -> dict[str, pd.DataFrame]:
    """The spread over the rotations (evaluation.report.rotation_spread) of each kind of world:
    metrics are each rotation's metrics table by its name, and kinds the kind of each rotation's
    worlds. The design's spread is over the three reported rotations, so the development
    rotation, whose worlds are smaller and tune the settings, is never pooled with them. A kind
    with fewer than two rotations has no spread."""
    by_kind: dict[str, dict[str, pd.DataFrame]] = {}
    for name, table in metrics.items():
        by_kind.setdefault(kinds[name], {})[name] = table
    return {kind: rotation_spread(held) for kind, held in by_kind.items() if len(held) > 1}


def evaluate_command(config: RunConfig, open_graph: Connect) -> None:
    del open_graph  # evaluation never queries TigerGraph
    rotations = [rotation_worlds(rotation, config) for rotation in config.worlds.rotations]
    for worlds in rotations:
        check_evaluable(worlds, config)
    device = _device()
    metrics: dict[str, pd.DataFrame] = {}
    for worlds in rotations:
        metrics[worlds.name] = _logged(
            results_dir(worlds.name),
            "evaluate",
            lambda worlds=worlds: evaluate_rotation(worlds, config, device),
        )
    kinds = {worlds.name: worlds.train[0].kind for worlds in rotations}
    for kind, spread in rotation_spreads(metrics, kinds).items():
        write_parquet(
            results_dir(ROTATIONS_RUN) / f"rotation_spread_{kind}.parquet",
            pa.Table.from_pandas(spread, preserve_index=False),
        )
        count = list(kinds.values()).count(kind)
        show(
            f"the spread over {plural(count, f'{kind} rotation')} is in "
            f"results/{ROTATIONS_RUN}/rotation_spread_{kind}.parquet"
        )


def score_command(config: RunConfig, open_graph: Connect) -> None:
    accounts, dates = config.score.accounts, config.score.dates
    if not accounts or not dates or not config.score.run:
        raise CommandError(
            "set score.accounts, score.dates and score.run in config.py: the accounts, the "
            "dates and the results folder whose model scores them"
        )
    folder = results_dir(config.score.run)
    paths = sorted(folder.glob(f"{ranker_name(None)}_seed*.pt"))
    if not paths:
        raise CommandError(f"{folder} holds no model file of the full model: run `mule train`")
    executor = open_graph()
    # No world is matched, so any loaded graph is scored, a bank's included; its rows go to a
    # scratch cache that this run alone reads, since a bank's graph changes between runs.
    check_queries(executor)
    show("ready: the installed queries are the repository's")
    by_time = {epoch_ms(day): list(accounts) for day in dates}
    with tempfile.TemporaryDirectory(prefix="mule-score-") as scratch:
        cache = Cache(Path(scratch))
        fetch(executor, cache, by_time, config.transport)
        device = _device()
        lock = gnn_lock(None, config.model, config.subgraph, config.training)
        saved = best_seed(paths, lock, device)
        keys = pd.DataFrame(
            [(t_ms, account) for t_ms, held in by_time.items() for account in held],
            columns=["t_ms", "account_id"],
        )
        scored = score(
            saved,
            ScoringBatches(cache, SCORED_WORLD, keys, config.training.batch_size),
            None,
            device,
        )
    scored["date"] = [datetime.fromtimestamp(t_ms / 1000, UTC).date() for t_ms in scored["t_ms"]]
    write_parquet(folder / SCORES_FILE, pa.Table.from_pandas(scored, preserve_index=False))
    show(f"scored with the seed {saved.seed} model, selection score {saved.score:.4f}")
    for row in scored.itertuples():
        show(
            f"  {row.date}  {row.account_id}  logit {row.logit:.4f}, rank {row.rank} of {len(accounts)}"
        )


def bench_command(config: RunConfig, open_graph: Connect) -> None:
    executor = open_graph()
    check_ready(executor, config)
    loaded, table, path = bench.run(executor, config)
    for line in bench.lines(table):
        show(line)
    show(f"the bench table of {loaded.config.name} is in {path}")


# The command line.

RUNNERS: dict[str, Callable[[RunConfig, Connect], None]] = {
    "install": install_command,
    "parity": parity_command,
    "prepare": prepare_command,
    "fetch": fetch_command,
    "audit": audit_command,
    "train": train_command,
    "evaluate": evaluate_command,
    "score": score_command,
    "bench": bench_command,
}

# What `mule --help` says: the module's docstring is for a reader of the code.
DESCRIPTION = (
    "Find money-mule accounts with a temporal graph neural network on TigerGraph. The commands "
    "take no options: worlds, roles, dates and settings live in config.py. See README.md."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mule", description=DESCRIPTION)
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    for name, summary in COMMANDS.items():
        commands.add_parser(name, help=summary, description=summary, add_help=False)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run a command of the command line. A command that stops for a reason a person can act on
    ends with its message and status 1."""
    command: str = build_parser().parse_args(argv).command
    # The events go to results/bench/ for the benchmark, else to results/commands/.
    events = bench_dir() if command == "bench" else results_dir(COMMANDS_RUN)
    try:
        _logged(events, command, lambda: RUNNERS[command](DEFAULT_CONFIG, connect))
    except STOPS as error:
        sys.exit(f"mule {command} stopped: {error}")
