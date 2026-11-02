"""The scored population at a scoring time: every page of list_accounts, cached once in the
world's folder of that time and reparsed by list_accounts' parser (docs/design.md, "The scored
population" and "Cache")."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mule_pattern_learner.data.population import (
    POPULATION_FILE,
    list_population,
    population,
    population_file,
    read_population,
)
from mule_pattern_learner.testing.calendar import ORDINARY, Spy, calendar_world
from mule_pattern_learner.testing.world import DAY_MS, T
from mule_pattern_learner.tigergraph.queries.list_accounts import ScorableAccount

# The scorable accounts of the calendar world at T: twelve ordinary customers' accounts and
# five mules' (a_gone closed 30 days before T; c_card is a credit card; x_far another bank's).
SCORABLE = sorted(
    [f"a_{name}" for name in ORDINARY] + ["a_long", "a_short", "a_edge", "a_early", "a_hidden"]
)


def test_every_page_is_read_until_one_comes_back_short() -> None:
    spy = Spy(calendar_world().graph())
    found = list_population(spy, T, page=5)
    assert [account.account_id for account in found] == SCORABLE
    assert all(account.owner_id == "p" + account.account_id[1:] for account in found)
    # 17 accounts in pages of 5: 5, 5, 5 and 2.
    afters = [params["after_id"] for _, params in spy.calls]
    assert spy.names() == ["list_accounts"] * 4
    assert afters == ["", SCORABLE[4], SCORABLE[9], SCORABLE[14]]
    # A page that ends the population exactly is followed by an empty one.
    spy.calls.clear()
    assert len(list_population(spy, T, page=17)) == 17 and len(spy.calls) == 2
    assert list_population(spy, T) == found


def test_the_population_is_cached_once_and_read_back(tmp_path: Path) -> None:
    spy = Spy(calendar_world().graph())
    path = population_file(tmp_path, T)
    assert path == tmp_path / "2024-07-01T00-00-00Z" / POPULATION_FILE
    first = population(spy, tmp_path, T)
    assert [account.account_id for account in first] == SCORABLE
    assert path.is_file() and [p.name for p in path.parent.iterdir()] == [POPULATION_FILE]
    calls = len(spy.calls)
    assert population(spy, tmp_path, T) == first and len(spy.calls) == calls
    assert read_population(path) == first
    # Another time is listed on its own: a week later a_gone is still closed.
    later = population(spy, tmp_path, T + 7 * DAY_MS)
    assert len(spy.calls) > calls and len(later) == 17


def test_a_cached_population_is_reparsed_with_the_live_checks(tmp_path: Path) -> None:
    path = tmp_path / POPULATION_FILE
    rows = {"account_id": ["a_1", "a_2"], "owner_id": ["p_1", ""]}
    pq.write_table(pa.table(rows), path)
    with pytest.raises(ValueError, match="a_2 without an owner"):
        read_population(path)
    pq.write_table(pa.table({"account_id": ["a_2", "a_1"], "owner_id": ["p_2", "p_1"]}), path)
    with pytest.raises(ValueError, match="strictly increasing id"):
        read_population(path)
    pq.write_table(pa.table({"account_id": ["a_1"], "owner_id": ["p_1"]}), path)
    assert read_population(path) == (ScorableAccount("a_1", "p_1"),)


def test_only_a_whole_second_has_a_folder(tmp_path: Path) -> None:
    for t_ms in (T + 1, T + 3_600_001, T - 999):
        with pytest.raises(ValueError, match="whole second"):
            population_file(tmp_path, t_ms)
