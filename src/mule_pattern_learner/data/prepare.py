"""mule prepare's data work on the loaded world (docs/design.md, "The pipeline" and "Commands"),
through the fetch executor, which runs read_world, list_accounts and read_labels and never the
answer key:

1. match the loaded world to a configured one by its identity file (worlds), and check that it
   holds the calendar's export window (calendar);
2. read the identified mules once (labels);
3. for every scored Monday, read the population from the world's cache or list and cache it
   (population), then find its positives, and, for each role the world plays in the
   configured rotations, its training draw, its selection keys and its test keys (examples);
4. weigh the training examples, report the world (report), refuse a selection world with too
   few identified active mules, and write the tables to the world's examples folder.

The selection and test keys of a 300,000-person world run to millions of rows, so they are
written a Monday at a time, and every table appears whole or not at all.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..config import DEFAULT_CONFIG, RunConfig
from ..paths import cache_dir
from ..tigergraph.executor import FetchExecutor, QueryExecutor
from ..tigergraph.world import cache_key
from .calendar import check_window, mondays
from .examples import (
    SELECTION_FILE,
    SELECTION_SCHEMA,
    TEST_FILE,
    TEST_SCHEMA,
    TRAINING_FILE,
    TRAINING_SCHEMA,
    Draw,
    TableWriter,
    Tally,
    examples_folder,
    examples_key,
    monday,
    scored_frame,
    selection_frame,
    training_frame,
    unlabelled,
)
from .labels import Labels, read_identified, unscored
from .population import population
from .report import REPORT_FILE, REPORT_SCHEMA, Report, build_report, refuse_selection
from .worlds import SELECT, TEST, TRAIN, LoadedWorld, find_world, roles


@dataclass(frozen=True)
class Prepared:
    """What mule prepare found and wrote: the loaded world, the roles it plays, its labels, its
    report, and the examples folder its tables are in."""

    world: LoadedWorld
    roles: frozenset[str]
    labels: Labels
    report: Report
    folder: Path


def prepare(
    executor: QueryExecutor,
    config: RunConfig = DEFAULT_CONFIG,
    identity_files: Sequence[Path] | None = None,
) -> Prepared:
    """Prepare the loaded world: identity_files are the identity files to match it among, by
    default every one in the records folder that .env names."""
    fetch = FetchExecutor(executor)
    loaded = find_world(fetch, config.worlds, identity_files)
    check_window(loaded.world, config.calendar)
    played = roles(loaded.config.name, config.worlds.rotations)
    labels = read_identified(fetch)
    root = cache_dir(loaded.identity) / cache_key(config.subgraph)
    key = examples_key(config.calendar, config.data)
    folder = examples_folder(loaded.identity, key, labels.identity)
    identified = labels.accounts
    writers: dict[str, TableWriter] = {}
    if SELECT in played:
        writers[SELECTION_FILE] = TableWriter(folder / SELECTION_FILE, SELECTION_SCHEMA)
    if TEST in played and config.data.scoring == "whole":
        writers[TEST_FILE] = TableWriter(folder / TEST_FILE, TEST_SCHEMA)
    tallies: list[Tally] = []
    draws: list[Draw] = []
    try:
        scored = mondays(config.calendar)
        for t_ms in scored:
            accounts = population(fetch, root, t_ms)
            day = monday(t_ms, [account.account_id for account in accounts], labels.mules)
            tallies.append(day.tally())
            if TRAIN in played:
                draws.append(unlabelled(day, identified, config.data))
            if SELECTION_FILE in writers:
                writers[SELECTION_FILE].write(selection_frame(day, identified, config.data))
            if TEST_FILE in writers:
                writers[TEST_FILE].write(scored_frame(day))
        positives = Counter(account for tally in tallies for account in tally.positives)
        training = training_frame(tallies, draws) if TRAIN in played else None
        missed = unscored(labels.mules, scored, positives)
        report = build_report(tallies, labels, missed, draws, training)
        if SELECT in played:
            refuse_selection(report, config.data.min_selection_identified)
        if training is not None:
            writers[TRAINING_FILE] = TableWriter(folder / TRAINING_FILE, TRAINING_SCHEMA)
            writers[TRAINING_FILE].write(training)
        writers[REPORT_FILE] = TableWriter(folder / REPORT_FILE, REPORT_SCHEMA)
        writers[REPORT_FILE].write(report.mondays)
    except BaseException:
        for writer in writers.values():
            writer.abort()
        raise
    for writer in writers.values():
        writer.commit()
    return Prepared(loaded, played, labels, report, folder)
