"""The cache of fetched rows: written once a part at a time, read back for exactly the keys asked,
reparsed with the live checks, and keyed by the world, the query versions and the settings
(docs/design.md, "Cache")."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mule_pattern_learner import paths
from mule_pattern_learner.config import SubgraphConfig
from mule_pattern_learner.subgraph.cache import (
    ACTIVITY,
    COMMUNITY,
    COMPLETE,
    COUNTERPARTIES,
    HUBS,
    INDEX_FILE,
    MONEY_FLOW,
    PAYMENT_EVENTS,
    SHARED_IDENTITY,
    TABLES,
    Cache,
    Table,
)
from mule_pattern_learner.testing.reference import ReferenceGraph
from mule_pattern_learner.testing.world import DAY_MS, T
from mule_pattern_learner.tigergraph.queries import (
    READ_HUBS,
    read_activity_change,
    read_community,
    read_counterparties,
    read_hubs,
    read_money_flow,
    read_payment_events,
    read_shared_identity,
)
from mule_pattern_learner.tigergraph.world import cache_key

IDENTITY = "0123456789abcdef0123456789abcdef"
HUBS_AT_T = ("a_shop",)
# Accounts of every kind of the world: look-alikes, a ring's mule and collector, a household,
# a ring sharing a device, a shopper, the hub itself and another bank's account.
ACCOUNTS = (
    "a_mule",
    "a_collector",
    "a_landlord",
    "a_split",
    "a_h1",
    "a_r1",
    "a_c000",
    "a_shop",
    "x_vic2",
)
PARAMS: dict[str, Callable[[Sequence[str]], dict[str, Any]]] = {
    MONEY_FLOW.name: lambda accounts: read_money_flow.params(T, accounts),
    COUNTERPARTIES.name: lambda accounts: read_counterparties.params(T, accounts, HUBS_AT_T),
    ACTIVITY.name: lambda accounts: read_activity_change.params(T, accounts),
    PAYMENT_EVENTS.name: lambda accounts: read_payment_events.params(T, accounts, HUBS_AT_T),
    SHARED_IDENTITY.name: lambda accounts: read_shared_identity.params(T, accounts, HUBS_AT_T),
    COMMUNITY.name: lambda accounts: read_community.params(T, accounts, HUBS_AT_T),
}


def _answer(graph: ReferenceGraph, table: Table[Any], accounts: Sequence[str]) -> Any:
    """The parsed answer of table's query for accounts at T, from the reference graph."""
    return table.parse(graph.run(table.query, PARAMS[table.name](accounts)), accounts)


@pytest.mark.parametrize("name", sorted(TABLES))
def test_each_table_reads_back_what_was_written(
    graph: ReferenceGraph, tmp_path: Path, name: str
) -> None:
    table = TABLES[name]
    cache = Cache(tmp_path)
    answer = _answer(graph, table, ACCOUNTS)
    cache.write(table, T, ACCOUNTS, [answer])
    assert cache.rows(table, T, ACCOUNTS) == answer
    folder = tmp_path / "2024-07-01" / name
    assert sorted(path.name for path in folder.iterdir()) == [INDEX_FILE, "part-0"]
    assert sorted(path.name for path in (folder / "part-0").iterdir()) == sorted(
        f"{kind.name}.parquet" for kind in table.kinds
    )


@pytest.mark.parametrize("name", sorted(TABLES))
def test_the_reader_gives_exactly_the_keys_asked(
    graph: ReferenceGraph, tmp_path: Path, name: str
) -> None:
    table = TABLES[name]
    cache = Cache(tmp_path)
    # Two requests' answers written as one part, then a third as another.
    first, second = ACCOUNTS[:4], ACCOUNTS[4:7]
    cache.write(
        table, T, first + second, [_answer(graph, table, first), _answer(graph, table, second)]
    )
    cache.write(table, T, ACCOUNTS[7:], [_answer(graph, table, ACCOUNTS[7:])])
    for asked in (["a_h1"], ["a_mule", "a_r1", "a_shop"], list(reversed(ACCOUNTS))):
        # An account's rows never depend on the others of its request.
        assert cache.rows(table, T, asked) == _answer(graph, table, sorted(asked))
    assert cache.missing(name, T, [*ACCOUNTS, "a_saver", "a_saver"]) == ["a_saver"]
    with pytest.raises(KeyError, match="lacks"):
        cache.rows(table, T, ["a_saver", "a_mule"])


def test_the_index_names_each_accounts_part(graph: ReferenceGraph, tmp_path: Path) -> None:
    cache = Cache(tmp_path)
    cache.write(COUNTERPARTIES, T, ["a_mule"], [_answer(graph, COUNTERPARTIES, ["a_mule"])])
    cache.write(COUNTERPARTIES, T, ["a_h1"], [_answer(graph, COUNTERPARTIES, ["a_h1"])])
    index = pq.read_table(tmp_path / "2024-07-01" / "counterparties" / INDEX_FILE)
    assert index.to_pylist() == [
        {"account_id": "a_mule", "part": 0},
        {"account_id": "a_h1", "part": 1},
    ]
    # A new reader of the same folder finds both.
    assert Cache(tmp_path).missing("counterparties", T, ["a_mule", "a_h1", "a_r1"]) == ["a_r1"]


def test_a_part_holds_only_the_accounts_it_was_written_for(
    graph: ReferenceGraph, tmp_path: Path
) -> None:
    cache = Cache(tmp_path)
    answer = _answer(graph, ACTIVITY, ["a_mule", "a_h1"])
    with pytest.raises(ValueError, match="rows of accounts it was not asked"):
        cache.write(ACTIVITY, T, ["a_mule"], [answer])
    with pytest.raises(ValueError, match="at least one account"):
        cache.write(ACTIVITY, T, [], [])


def test_a_crash_before_a_parts_rename_leaves_no_part(
    graph: ReferenceGraph, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = Cache(tmp_path)
    answer = _answer(graph, PAYMENT_EVENTS, ACCOUNTS)

    def crash(self: Path, target: Path) -> Path:
        raise OSError("the process stopped")

    monkeypatch.setattr(Path, "rename", crash)
    with pytest.raises(OSError, match="stopped"):
        cache.write(PAYMENT_EVENTS, T, ACCOUNTS, [answer])
    monkeypatch.undo()
    folder = tmp_path / "2024-07-01" / "payment_events"
    # Neither the part nor its temporary folder is left, and no account reads as cached.
    assert list(folder.iterdir()) == []
    assert cache.missing("payment_events", T, ACCOUNTS) == sorted(ACCOUNTS)
    cache.write(PAYMENT_EVENTS, T, ACCOUNTS, [answer])
    assert cache.rows(PAYMENT_EVENTS, T, ACCOUNTS) == answer


def test_a_crash_before_the_index_leaves_a_part_never_read(
    graph: ReferenceGraph, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = Cache(tmp_path)
    answer = _answer(graph, MONEY_FLOW, ACCOUNTS)

    def crash(self: Path, target: Path) -> Path:
        raise OSError("the process stopped")

    monkeypatch.setattr(Path, "replace", crash)
    with pytest.raises(OSError, match="stopped"):
        cache.write(MONEY_FLOW, T, ACCOUNTS, [answer])
    monkeypatch.undo()
    folder = tmp_path / "2024-07-01" / "money_flow"
    assert sorted(path.name for path in folder.iterdir()) == ["part-0"]
    reader = Cache(tmp_path)
    assert reader.missing("money_flow", T, ACCOUNTS) == sorted(ACCOUNTS)
    # The accounts are written again into a part of their own, which the index names.
    reader.write(MONEY_FLOW, T, ACCOUNTS, [answer])
    assert sorted(path.name for path in folder.iterdir()) == [INDEX_FILE, "part-0", "part-1"]
    assert reader.rows(MONEY_FLOW, T, ACCOUNTS) == answer
    index = pq.read_table(folder / INDEX_FILE)
    assert set(index["part"].to_pylist()) == {1}


def test_cached_rows_are_reparsed_with_the_live_checks(
    graph: ReferenceGraph, tmp_path: Path
) -> None:
    cache = Cache(tmp_path)
    cache.write(MONEY_FLOW, T, ["a_mule"], [_answer(graph, MONEY_FLOW, ["a_mule"])])
    path = tmp_path / "2024-07-01" / "money_flow" / "part-0" / "accounts.parquet"
    rows = pq.read_table(path)
    # A share of 2 the walk cannot give.
    planted = rows.set_column(
        rows.schema.get_field_index("left_15m"), "left_15m", pa.array([2.0], pa.float64())
    )
    pq.write_table(planted, path)
    with pytest.raises(ValueError, match=r"a share outside \[0, 1\]"):
        Cache(tmp_path).rows(MONEY_FLOW, T, ["a_mule"])


def test_two_subgraphs_that_disagree_on_a_shared_row_are_refused(
    graph: ReferenceGraph, tmp_path: Path
) -> None:
    cache = Cache(tmp_path)
    cache.write(
        PAYMENT_EVENTS,
        T,
        ["a_mule", "a_collector"],
        [_answer(graph, PAYMENT_EVENTS, ["a_mule", "a_collector"])],
    )
    path = tmp_path / "2024-07-01" / "payment_events" / "part-0" / "payments.parquet"
    rows = pq.read_table(path).to_pylist()
    # z_mule_f1, from a_mule to a_collector, is in both subgraphs; one copy changes its amount.
    copies = [row for row in rows if row["payment_id"] == "z_mule_f1"]
    assert len(copies) == 2
    copies[0]["amount"] += 1
    pq.write_table(pa.Table.from_pylist(rows, schema=pq.read_schema(path)), path)
    assert len(cache.rows(PAYMENT_EVENTS, T, ["a_mule"])) == 1
    with pytest.raises(ValueError, match="two rows of z_mule_f1 that differ"):
        cache.rows(PAYMENT_EVENTS, T, ["a_mule", "a_collector"])


def test_the_hubs_of_a_monday_are_cached_once(graph: ReferenceGraph, tmp_path: Path) -> None:
    cache = Cache(tmp_path)
    assert cache.hubs(T) is None
    hubs = read_hubs.parse(graph.run(READ_HUBS, read_hubs.params(T)))
    assert [account.account_id for account in hubs.accounts] == list(HUBS_AT_T)
    cache.write_hubs(T, hubs)
    assert cache.hubs(T) == hubs == Cache(tmp_path).hubs(T)
    with pytest.raises(ValueError, match="cached already"):
        cache.write_hubs(T, hubs)
    assert HUBS.name == "hubs" and (tmp_path / "2024-07-01" / "hubs" / "part-0").is_dir()


def test_complete_subgraphs_are_an_index_alone(tmp_path: Path) -> None:
    cache = Cache(tmp_path)
    cache.complete(T, ["a_mule", "a_h1"])
    assert cache.missing(COMPLETE, T, ["a_h1", "a_r1"]) == ["a_r1"]
    assert [path.name for path in (tmp_path / "2024-07-01" / COMPLETE).iterdir()] == [INDEX_FILE]


def test_only_a_monday_at_midnight_has_a_folder(tmp_path: Path) -> None:
    cache = Cache(tmp_path)
    for t_ms in (T + 1, T + DAY_MS, T - 3_600_000):
        with pytest.raises(ValueError, match="Monday at 00:00 UTC"):
            cache.missing("activity", t_ms, ["a_mule"])
        with pytest.raises(ValueError, match="Monday at 00:00 UTC"):
            cache.hubs(t_ms)


def test_two_settings_never_share_a_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path)
    found = Cache.of(IDENTITY, SubgraphConfig())
    assert found.root == tmp_path / "cache" / IDENTITY / cache_key(SubgraphConfig())
    other = Cache.of(IDENTITY, SubgraphConfig(crowd_threshold=9))
    assert other.root != found.root and other.root.parent == found.root.parent
    world = Cache.of("f" * 32, SubgraphConfig())
    assert world.root.parent != found.root.parent
