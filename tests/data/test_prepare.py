"""mule prepare's data work on the calendar world through a spy executor backed by the reference
graph: it runs read_world, read_labels and list_accounts and nothing else, caches each Monday's
population once, and writes the examples of each role the world plays (docs/design.md, "The
pipeline" and "Worlds, examples and weights")."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from mule_pattern_learner.config import DEFAULT_CONFIG, RunConfig
from mule_pattern_learner.data.examples import (
    POSITIVE,
    SELECTION_FILE,
    SELECTION_SCHEMA,
    TEST_FILE,
    TEST_SCHEMA,
    TRAINING_FILE,
    TRAINING_SCHEMA,
    examples_key,
    read_table,
)
from mule_pattern_learner.data.population import POPULATION_FILE
from mule_pattern_learner.data.prepare import Prepared, prepare
from mule_pattern_learner.data.report import REPORT_FILE
from mule_pattern_learner.data.worlds import SELECT, TEST, TRAIN
from mule_pattern_learner.testing.calendar import Spy, calendar_world, identity_text, world_of
from mule_pattern_learner.testing.world import DAY_MS, T
from mule_pattern_learner.tigergraph.executor import FetchExecutor, QueryNotAllowed
from mule_pattern_learner.tigergraph.world import cache_key, identity

Records = Callable[[str, str], Path]
# Selection needs 3 identified active mules here: the calendar world has a_long, a_short and
# a_edge.
CONFIG = replace(
    DEFAULT_CONFIG,
    data=replace(DEFAULT_CONFIG.data, per_positive=2, min_unlabelled=3, min_selection_identified=3),
)
W = 7 * DAY_MS


def _prepared(records: Records, seed: int, config: RunConfig = CONFIG) -> tuple[Spy, Prepared]:
    graph = calendar_world().graph()
    path = records(f"world_{seed}.identity.toml", identity_text(world_of(graph), seed))
    spy = Spy(graph)
    return spy, prepare(spy, config, [path])


def test_a_world_that_plays_every_role(records: Records, data_dir: Path) -> None:
    spy, prepared = _prepared(records, 2001)
    assert prepared.world.config.name == "A" and prepared.roles == {TRAIN, SELECT, TEST}
    world_identity = identity(prepared.world.world)
    key = examples_key(CONFIG.calendar, CONFIG.data)
    assert prepared.folder == (
        data_dir / "examples" / world_identity / key / prepared.labels.identity
    )
    assert sorted(path.name for path in prepared.folder.iterdir()) == sorted(
        [TRAINING_FILE, SELECTION_FILE, TEST_FILE, REPORT_FILE]
    )
    # The data code ran these and nothing else: read_world once, read_labels once, and one page
    # of list_accounts per scored Monday.
    assert spy.names() == ["read_world", "read_labels"] + ["list_accounts"] * 52
    training = read_table(prepared.folder / TRAINING_FILE, TRAINING_SCHEMA)
    positives = training[training["label"] == POSITIVE]
    assert sorted(set(positives["account_id"])) == ["a_edge", "a_long", "a_short"]
    assert len(training) == 28 + 157
    selection = read_table(prepared.folder / SELECTION_FILE, SELECTION_SCHEMA)
    # 18 scorable accounts on the 9 Mondays before a_gone closed, 17 on the other 43.
    assert len(selection) == 9 * 18 + 43 * 17
    assert selection["positive"].sum() == 28
    test = read_table(prepared.folder / TEST_FILE, TEST_SCHEMA)
    assert len(test) == len(selection) and set(test["t_ms"]) == set(CONFIG.calendar.mondays())
    report = prepared.report
    assert report.identified == 5 and report.identified_active == 3
    assert report.unscored.no_active_monday == ("a_early",)
    assert report.unscored.never_scorable == ("a_gone",)
    # Each Monday's population is in the world's cache, in the Monday's folder.
    population = data_dir / "cache" / world_identity / cache_key(CONFIG.subgraph)
    assert (population / "2024-07-01" / POPULATION_FILE).is_file()
    assert len(list(population.iterdir())) == 52


def test_a_second_prepare_reads_the_cached_populations(records: Records, data_dir: Path) -> None:
    graph = calendar_world().graph()
    path = records("world_1001.identity.toml", identity_text(world_of(graph), 1001))
    first = prepare(Spy(graph), CONFIG, [path])
    spy = Spy(graph)
    second = prepare(spy, CONFIG, [path])
    assert spy.names() == ["read_world", "read_labels"]
    assert second.report.mondays.equals(first.report.mondays)
    training = read_table(second.folder / TRAINING_FILE, TRAINING_SCHEMA)
    assert training.equals(read_table(first.folder / TRAINING_FILE, TRAINING_SCHEMA))
    assert data_dir.is_dir()


@pytest.mark.parametrize(
    ("seed", "scoring", "tables"),
    [
        (1001, "whole", [REPORT_FILE, TRAINING_FILE]),
        (1002, "whole", [REPORT_FILE, SELECTION_FILE]),
        (1002, "fallback", [REPORT_FILE, SELECTION_FILE]),
        (1003, "whole", [REPORT_FILE, TEST_FILE]),
        # Under the fallback, mule audit builds the test world's list, from the truth.
        (1003, "fallback", [REPORT_FILE]),
        # The 5,000-person test world plays no role.
        (3001, "whole", [REPORT_FILE]),
    ],
)
def test_each_world_writes_the_tables_of_its_roles(
    records: Records, data_dir: Path, seed: int, scoring: str, tables: list[str]
) -> None:
    config = replace(CONFIG, data=replace(CONFIG.data, scoring=scoring))
    _, prepared = _prepared(records, seed, config)
    assert sorted(path.name for path in prepared.folder.iterdir()) == tables
    assert prepared.folder.is_relative_to(data_dir)


def test_a_selection_world_with_too_few_identified_active_mules_is_refused(
    records: Records, data_dir: Path
) -> None:
    with pytest.raises(ValueError, match="3 identified active mules, fewer than the 5"):
        _prepared(records, 1002, DEFAULT_CONFIG)
    # Nothing of the examples was written; the populations stay cached.
    examples = data_dir / "examples"
    assert not [path for path in examples.rglob("*") if path.is_file()]
    assert len(list((data_dir / "cache").rglob(POPULATION_FILE))) == 52


def test_a_world_without_its_window_or_its_identity_file_is_refused(
    records: Records, data_dir: Path
) -> None:
    w = calendar_world()
    w.pay("x_late", "a_n03", "x_far", T + 50 * W, 5.0, "ach")
    graph = w.graph()
    path = records("world_1001.identity.toml", identity_text(world_of(graph), 1001))
    with pytest.raises(ValueError, match="last payment"):
        prepare(Spy(graph), CONFIG, [path])
    with pytest.raises(ValueError, match="No identity file of 0"):
        prepare(Spy(graph), CONFIG, [])
    assert not data_dir.exists()


def test_the_data_code_runs_through_the_fetch_executor() -> None:
    spy = Spy(calendar_world().graph())
    with pytest.raises(QueryNotAllowed, match="does not run read_answer_key"):
        FetchExecutor(spy).run("read_answer_key", {"after_id": "", "page": 1})
    with pytest.raises(QueryNotAllowed, match="does not run profile_hubs"):
        FetchExecutor(spy).run("profile_hubs", {"t_ms": T})
    assert spy.calls == []
