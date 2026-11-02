"""The cache of fetched rows: written once a part at a time, each part with its own index, read
back for exactly the keys asked, reparsed with the live checks, its indexes held for one scoring
time at a time, and keyed by the world, the query versions and the settings (docs/design.md,
"Cache")."""

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
    CROWDS,
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
from mule_pattern_learner.tigergraph.queries.mark_crowds import Crowd, Crowds
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
    folder = tmp_path / "2024-07-01T00-00-00Z" / name
    assert [path.name for path in folder.iterdir()] == ["part-0"]
    # The part holds a file of each kind of row, and its index of the accounts.
    assert sorted(path.name for path in (folder / "part-0").iterdir()) == sorted(
        [INDEX_FILE, *(f"{kind.name}.parquet" for kind in table.kinds)]
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


def test_each_part_holds_the_index_of_its_accounts(graph: ReferenceGraph, tmp_path: Path) -> None:
    cache = Cache(tmp_path)
    cache.write(COUNTERPARTIES, T, ["a_mule"], [_answer(graph, COUNTERPARTIES, ["a_mule"])])
    cache.write(
        COUNTERPARTIES,
        T,
        ["a_r1", "a_h1"],
        [_answer(graph, COUNTERPARTIES, ["a_h1", "a_r1"])],
    )
    folder = tmp_path / "2024-07-01T00-00-00Z" / "counterparties"
    for part, accounts in ((0, ["a_mule"]), (1, ["a_h1", "a_r1"])):
        index = pq.read_table(folder / f"part-{part}" / INDEX_FILE)
        assert index.to_pylist() == [{"account_id": account} for account in accounts]
    # A new reader of the same folder finds both parts' accounts.
    assert Cache(tmp_path).missing("counterparties", T, ["a_mule", "a_h1", "a_r1", "a_c000"]) == [
        "a_c000"
    ]


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
    folder = tmp_path / "2024-07-01T00-00-00Z" / "payment_events"
    # Neither the part nor its temporary folder is left, and no account reads as cached.
    assert list(folder.iterdir()) == []
    assert cache.missing("payment_events", T, ACCOUNTS) == sorted(ACCOUNTS)
    cache.write(PAYMENT_EVENTS, T, ACCOUNTS, [answer])
    assert cache.rows(PAYMENT_EVENTS, T, ACCOUNTS) == answer


def test_a_crash_while_a_part_is_written_leaves_no_part(
    graph: ReferenceGraph, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = Cache(tmp_path)
    answer = _answer(graph, MONEY_FLOW, ACCOUNTS)
    written: list[Path] = []
    write_table = pq.write_table

    def crash_at_the_index(table: pa.Table, where: Path, *args: Any, **keywords: Any) -> None:
        if Path(where).name == INDEX_FILE:
            raise OSError("the process stopped")
        written.append(Path(where))
        write_table(table, where, *args, **keywords)

    monkeypatch.setattr(pq, "write_table", crash_at_the_index)
    with pytest.raises(OSError, match="stopped"):
        cache.write(MONEY_FLOW, T, ACCOUNTS, [answer])
    monkeypatch.undo()
    # The rows of the part were written, into a temporary folder that is gone; with no index
    # in place no account reads as cached, in this reader or a new one.
    assert len(written) == len(MONEY_FLOW.kinds)
    folder = tmp_path / "2024-07-01T00-00-00Z" / "money_flow"
    assert list(folder.iterdir()) == []
    for reader in (cache, Cache(tmp_path)):
        assert reader.missing("money_flow", T, ACCOUNTS) == sorted(ACCOUNTS)
    cache.write(MONEY_FLOW, T, ACCOUNTS, [answer])
    assert [path.name for path in folder.iterdir()] == ["part-0"]
    assert Cache(tmp_path).rows(MONEY_FLOW, T, ACCOUNTS) == answer


def test_cached_rows_are_reparsed_with_the_live_checks(
    graph: ReferenceGraph, tmp_path: Path
) -> None:
    cache = Cache(tmp_path)
    cache.write(MONEY_FLOW, T, ["a_mule"], [_answer(graph, MONEY_FLOW, ["a_mule"])])
    path = tmp_path / "2024-07-01T00-00-00Z" / "money_flow" / "part-0" / "accounts.parquet"
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
    path = tmp_path / "2024-07-01T00-00-00Z" / "payment_events" / "part-0" / "payments.parquet"
    rows = pq.read_table(path).to_pylist()
    # z_mule_f1, from a_mule to a_collector, is in both subgraphs; one copy changes its amount.
    copies = [row for row in rows if row["payment_id"] == "z_mule_f1"]
    assert len(copies) == 2
    copies[0]["amount"] += 1
    pq.write_table(pa.Table.from_pylist(rows, schema=pq.read_schema(path)), path)
    assert len(cache.rows(PAYMENT_EVENTS, T, ["a_mule"])) == 1
    with pytest.raises(ValueError, match="two rows of z_mule_f1 that differ"):
        cache.rows(PAYMENT_EVENTS, T, ["a_mule", "a_collector"])


def test_the_hubs_of_a_scoring_time_are_cached_once(graph: ReferenceGraph, tmp_path: Path) -> None:
    cache = Cache(tmp_path)
    assert cache.hubs(T) is None
    hubs = read_hubs.parse(graph.run(READ_HUBS, read_hubs.params(T)))
    assert [account.account_id for account in hubs.accounts] == list(HUBS_AT_T)
    cache.write_hubs(T, hubs)
    assert cache.hubs(T) == hubs == Cache(tmp_path).hubs(T)
    with pytest.raises(ValueError, match="cached already"):
        cache.write_hubs(T, hubs)
    assert HUBS.name == "hubs" and (tmp_path / "2024-07-01T00-00-00Z" / "hubs" / "part-0").is_dir()


def test_the_crowds_of_a_scoring_time_are_cached_once(tmp_path: Path) -> None:
    cache = Cache(tmp_path)
    assert cache.crowds(T) is None
    crowds = Crowds((Crowd("d_kiosk", 9, True, False),), (Crowd("i_carrier", 201, False, True),))
    cache.write_crowds(T, crowds)
    assert cache.crowds(T) == crowds == Cache(tmp_path).crowds(T)
    with pytest.raises(ValueError, match="crowds of 2024-07-01T00-00-00Z are cached already"):
        cache.write_crowds(T, crowds)
    part = tmp_path / "2024-07-01T00-00-00Z" / "crowds" / "part-0"
    assert CROWDS.name == "crowds" and part.is_dir()
    # Read back, they are reparsed with mark_crowds' checks: a crowd of 8 customers is none.
    path = part / "devices.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[0]["customers_90d"] = 8
    pq.write_table(pa.Table.from_pylist(rows, schema=pq.read_schema(path)), path)
    with pytest.raises(ValueError, match="among its devices"):
        Cache(tmp_path).crowds(T)


def test_complete_subgraphs_are_parts_of_an_index_alone(tmp_path: Path) -> None:
    cache = Cache(tmp_path)
    cache.complete(T, ["a_mule", "a_h1"])
    cache.complete(T, ["a_r1"])
    assert cache.missing(COMPLETE, T, ["a_h1", "a_r1", "a_c000"]) == ["a_c000"]
    folder = tmp_path / "2024-07-01T00-00-00Z" / COMPLETE
    assert sorted(path.name for path in folder.iterdir()) == ["part-0", "part-1"]
    for part in ("part-0", "part-1"):
        assert [path.name for path in (folder / part).iterdir()] == [INDEX_FILE]
    with pytest.raises(ValueError, match="at least one account"):
        cache.complete(T, [])


def test_only_a_whole_second_has_a_folder(tmp_path: Path) -> None:
    cache = Cache(tmp_path)
    for t_ms in (T + 1, T + 3_600_001, T - 999):
        with pytest.raises(ValueError, match="whole second"):
            cache.missing("activity", t_ms, ["a_mule"])
        with pytest.raises(ValueError, match="whole second"):
            cache.hubs(t_ms)
        with pytest.raises(ValueError, match="whole second"):
            cache.crowds(t_ms)
        with pytest.raises(ValueError, match="whole second"):
            cache.forget(t_ms)


def test_two_times_of_one_week_have_folders_of_their_own(tmp_path: Path) -> None:
    cache = Cache(tmp_path)
    # Thursday 2024-07-04 at 13:47:21, inside the week that starts at T.
    inside = T + 3 * DAY_MS + 49_641_000
    cache.complete(T, ["a_mule"])
    cache.complete(inside, ["a_h1"])
    assert (tmp_path / "2024-07-01T00-00-00Z" / COMPLETE).is_dir()
    assert (tmp_path / "2024-07-04T13-47-21Z" / COMPLETE).is_dir()
    assert cache.missing(COMPLETE, T, ["a_mule", "a_h1"]) == ["a_h1"]
    assert cache.missing(COMPLETE, inside, ["a_mule", "a_h1"]) == ["a_mule"]


def test_the_indexes_of_one_scoring_time_are_held_at_a_time(
    tmp_path: Path, index_reads: list[Path]
) -> None:
    """The indexes of a scoring time run to millions of entries at 300,000 people, so a cache that
    goes through 52 weeks holds one time's and reads another's again when it is asked."""
    later = T + 7 * DAY_MS
    cache = Cache(tmp_path)
    cache.complete(T, ["a_mule"])
    cache.complete(later, ["a_h1"])

    def reading(t_ms: int, accounts: list[str], expected: list[str]) -> int:
        """How many indexes asking about accounts at the time t reads, and what it lacks."""
        before = len(index_reads)
        assert cache.missing(COMPLETE, t_ms, accounts) == expected
        return len(index_reads) - before

    # The time last asked about is held: asking again reads nothing.
    assert reading(later, ["a_h1", "a_mule"], ["a_mule"]) == 0
    assert reading(later, ["a_h1"], []) == 0
    # Asking about the other drops it, and the first is read again when it is asked about.
    assert reading(T, ["a_h1", "a_mule"], ["a_h1"]) == 1
    assert reading(T, ["a_mule"], []) == 0
    assert reading(later, ["a_h1"], []) == 1
    # forget drops the time it names, and no other.
    cache.forget(T)
    assert reading(later, ["a_h1"], []) == 0
    cache.forget(later)
    assert reading(later, ["a_h1"], []) == 1
    assert reading(later, ["a_h1"], []) == 0
    # What a time holds is read again from its parts, new parts included.
    cache.complete(later, ["a_r1"])
    cache.forget(later)
    assert reading(later, ["a_h1", "a_r1"], []) == 2


def test_a_reader_of_many_times_holds_the_indexes_of_one(
    graph: ReferenceGraph, tmp_path: Path, index_reads: list[Path]
) -> None:
    cache = Cache(tmp_path)
    times = [T + week * 7 * DAY_MS for week in range(4)]
    answer = _answer(graph, ACTIVITY, ["a_mule", "a_h1"])
    for t_ms in times:
        cache.write(ACTIVITY, t_ms, ["a_mule", "a_h1"], [answer])
    # The last time written is held. Reading each week in turn, twice, reads its index
    # once, the last time's included since the first time dropped it; going back to the
    # first reads it again, since only the time last read is held.
    before = len(index_reads)
    for t_ms in times:
        assert cache.rows(ACTIVITY, t_ms, ["a_mule"]) == _answer(graph, ACTIVITY, ["a_mule"])
        assert cache.rows(ACTIVITY, t_ms, ["a_h1"]) == _answer(graph, ACTIVITY, ["a_h1"])
    assert len(index_reads) - before == len(times)
    cache.rows(ACTIVITY, times[0], ["a_mule"])
    assert len(index_reads) - before == len(times) + 1


def test_two_settings_never_share_a_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path)
    found = Cache.of(IDENTITY, SubgraphConfig())
    assert found.root == tmp_path / "cache" / IDENTITY / cache_key(SubgraphConfig())
    other = Cache.of(IDENTITY, SubgraphConfig(crowd_threshold=9))
    assert other.root != found.root and other.root.parent == found.root.parent
    world = Cache.of("f" * 32, SubgraphConfig())
    assert world.root.parent != found.root.parent
