"""mule parity: the live graph and the reference graph on the same export, compared (docs/design.md,
"Commands").

The live graph is stood in for by a reference graph behind a spy executor, which a test may make
answer otherwise than the reference does (a tamper), and the export is written here in the
loader's format (testing.shards.write_export). The world is testing.fixtures.batch_world, whose
mules (a_collector, a_mule, a_mule2, a_mule3, a_r1 and a_r2) are the true mules the sample
oversamples, and whose 224 scorable accounts of T, 2024-07-01, and of T + 1 week give a sample of
24 accounts on each of two Mondays: 6 mules, a quarter of 24, and 18 others, in 3 requests of 8.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields, replace
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from mule_pattern_learner.config import (
    DEFAULT_CONFIG,
    CalendarConfig,
    ParityConfig,
    RunConfig,
)
from mule_pattern_learner.subgraph.cache import MONEY_FLOW
from mule_pattern_learner.testing import fixtures
from mule_pattern_learner.testing.calendar import Spy, world_of
from mule_pattern_learner.testing.parity import (
    WORK_COUNTERS,
    Communities,
    Failure,
    ParityError,
    _canonical,  # pyright: ignore[reportPrivateUsage]
    _Comparison,  # pyright: ignore[reportPrivateUsage]
    differences,
    find_export,
    run,
    run_on,
    sample_accounts,
    sample_mondays,
)
from mule_pattern_learner.testing.reference import ReferenceGraph
from mule_pattern_learner.testing.shards import write_export
from mule_pattern_learner.testing.world import DAY_MS, T, WorldBuilder
from mule_pattern_learner.tigergraph.executor import ParityExecutor
from mule_pattern_learner.tigergraph.installer import parity_versions
from mule_pattern_learner.tigergraph.queries import (
    LIST_ACCOUNTS,
    PROFILE_HUBS,
    QUERY_FILES,
    READ_ACTIVITY_CHANGE,
    READ_ANSWER_KEY,
    READ_COMMUNITY,
    READ_COUNTERPARTIES,
    READ_HUBS,
    READ_LABELS,
    READ_MONEY_FLOW,
    READ_PAYMENT_EVENTS,
    READ_SHARED_IDENTITY,
    READ_WORLD,
    VERSIONS,
    read_money_flow,
)
from mule_pattern_learner.tigergraph.queries.read_money_flow import MoneyFlows

W = 7 * DAY_MS
MULES = {"a_collector", "a_mule", "a_mule2", "a_mule3", "a_r1", "a_r2"}
# Two scored Mondays, T and T + 1 week; 24 accounts on each in requests of 8, two of them
# fetched alone.
CONFIG = replace(
    DEFAULT_CONFIG,
    calendar=CalendarConfig(history_days=182, scored_mondays=2),
    parity=ParityConfig(mondays=2, accounts=24, together=8, alone=2),
)
TOLERANCE = CONFIG.parity.tolerance
Tamper = Callable[[str, dict[str, Any], list[dict[str, Any]]], list[dict[str, Any]]]


class Live(Spy):
    """The live graph as a connection executor: a spy over a reference graph, whose answers a
    tamper may change, and a connection whose server is TigerGraph 4.2.5."""

    graph_name = "Mule_Pattern_Learner"
    client: Any = None

    def __init__(self, graph: ReferenceGraph, tamper: Tamper | None = None) -> None:
        super().__init__(graph)
        self.tamper = tamper

    def run(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        name: str,
        params: dict[str, Any],
        *,
        timeout_s: float | None = None,
        attempts: int | None = None,
        timeout_retries: int = 1,
    ) -> list[dict[str, Any]]:
        rows = super().run(
            name, params, timeout_s=timeout_s, attempts=attempts, timeout_retries=timeout_retries
        )
        return rows if self.tamper is None else self.tamper(name, params, rows)

    def call[T](
        self, operation: Callable[[Any], T], *, what: str, attempts: int | None = None
    ) -> T:
        class Connection:
            def getVer(self) -> str:  # noqa: N802
                return "4.2.5"

        return operation(Connection())

    def gsql(self, text: str, *, what: str = "gsql", attempts: int | None = None) -> str:
        raise AssertionError("parity runs no GSQL statement")


@pytest.fixture(scope="module")
def tables() -> dict[str, pd.DataFrame]:
    return fixtures.batch_world().tables()


@pytest.fixture(scope="module")
def graph(tables: dict[str, pd.DataFrame]) -> ReferenceGraph:
    return ReferenceGraph(tables)


@pytest.fixture(scope="module")
def exports(tables: dict[str, pd.DataFrame], tmp_path_factory: pytest.TempPathFactory) -> Path:
    """An export root holding the world's export, in a folder of its own."""
    root = tmp_path_factory.mktemp("exports")
    write_export(root / "mule_temporal_0123456789ab", tables)
    return root


def _blocks(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    """The printed block of a query's answer that holds key."""
    return next(row[key] for row in rows if key in row)


def test_the_same_graph_on_both_sides_passes_every_query(
    graph: ReferenceGraph, exports: Path, tmp_path: Path
) -> None:
    record = tmp_path / "parity.toml"
    found = run(Live(graph), CONFIG, exports=exports, record=record, today=date(2026, 10, 24))
    assert found.failed == () and found.passed == tuple(QUERY_FILES)
    assert found.tigergraph == "4.2.5" and found.export == exports / "mule_temporal_0123456789ab"
    assert found.mondays == (T, T + W)
    # read_world, read_labels and read_answer_key are read whole: once, and the pages and
    # refusals of the last two (a page of 0 and a page above the limit).
    checks = {name: result.checks for name, result in found.queries.items()}
    assert (checks[READ_WORLD], checks[READ_LABELS], checks[READ_ANSWER_KEY]) == (1, 3, 3)
    # On each Monday: every page of the scorable accounts and 3 refusals; the hubs and the
    # profile, and a refusal each.
    assert (checks[LIST_ACCOUNTS], checks[READ_HUBS], checks[PROFILE_HUBS]) == (8, 4, 4)
    # B1 and B3: 3 requests of 8, 2 accounts alone and 4 refusals (a t that is no Monday, no
    # account, a repeated and an unknown account: this world has no account that opens after
    # t) on each Monday; B2, B4 and B6 the same; B5 also checks its leaf flags once a Monday.
    for name in (READ_MONEY_FLOW, READ_ACTIVITY_CHANGE, READ_COUNTERPARTIES):
        assert checks[name] == 2 * (3 + 2 + 4)
    assert checks[READ_PAYMENT_EVENTS] == checks[READ_COMMUNITY] == 2 * (3 + 2 + 4)
    assert checks[READ_SHARED_IDENTITY] == 2 * (3 + 2 + 4 + 1)
    # B6's sample: 48 communities; the world has no crowd, so none touches a leaf, which the
    # report says.
    assert found.communities == Communities(48, 0, 0)
    assert any("met no crowd" in line for line in found.lines())
    assert all(result.mismatches == [] for result in found.queries.values())
    # The record holds every version, as the parity gate reads it.
    assert parity_versions(record) == set(VERSIONS.items())
    assert (
        'tigergraph = "4.2.5"' in record.read_text() and "date = 2026-10-24" in record.read_text()
    )


def test_a_double_within_the_tolerance_is_the_same_and_beyond_it_is_not(
    graph: ReferenceGraph, exports: Path, tmp_path: Path
) -> None:
    def scaled(factor: float) -> Tamper:
        def tamper(
            name: str, params: dict[str, Any], rows: list[dict[str, Any]]
        ) -> list[dict[str, Any]]:
            if name == READ_MONEY_FLOW and "t_ms" in params and rows[0].get("status") == "ok":
                for vertex in _blocks(rows, "accounts"):
                    vertex["attributes"]["left_1d"] *= factor
            return rows

        return tamper

    near = run(
        Live(graph, scaled(1 + TOLERANCE / 10)), CONFIG, exports=exports, record=tmp_path / "a.toml"
    )
    assert near.failed == ()
    far = run(Live(graph, scaled(1 + 1e-9)), CONFIG, exports=exports, record=tmp_path / "b.toml")
    assert far.failed == (READ_MONEY_FLOW,)
    result = far.queries[READ_MONEY_FLOW]
    assert result.mismatched >= 1 and "left_1d" in result.mismatches[0].what
    # The query that failed has no entry in the record, and the others have.
    versions = parity_versions(tmp_path / "b.toml")
    assert (READ_MONEY_FLOW, VERSIONS[READ_MONEY_FLOW]) not in versions
    assert len(versions) == len(QUERY_FILES) - 1


def test_an_integer_is_exact_where_a_double_has_a_tolerance(
    graph: ReferenceGraph, exports: Path, tmp_path: Path
) -> None:
    def tamper(
        name: str, params: dict[str, Any], rows: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        # B1 counts one cent more than the reference in the 90 days.
        if name == READ_MONEY_FLOW and rows[0].get("status") == "ok":
            for vertex in _blocks(rows, "accounts"):
                vertex["attributes"]["in_amount_90d"] += 1
        return rows

    found = run(Live(graph, tamper), CONFIG, exports=exports, record=tmp_path / "parity.toml")
    assert found.failed == (READ_MONEY_FLOW,)
    assert "in_amount_90d" in found.queries[READ_MONEY_FLOW].mismatches[0].what


def _steps(name: str, params: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Live B1 prints the steps of its walk, positive and the same for an account however the
    request is cut; the reference prints 0 (docs/design.md, "Query plumbing")."""
    if name == READ_MONEY_FLOW and rows[0].get("status") == "ok":
        for vertex in _blocks(rows, "accounts"):
            vertex["attributes"]["scan_steps"] = 1 + len(vertex["attributes"]["account_id"])
    return rows


def test_a_work_counter_is_not_compared(
    graph: ReferenceGraph, exports: Path, tmp_path: Path
) -> None:
    record = tmp_path / "parity.toml"
    found = run(Live(graph, _steps), CONFIG, exports=exports, record=record)
    assert found.failed == ()
    assert (READ_MONEY_FLOW, VERSIONS[READ_MONEY_FLOW]) in parity_versions(record)


def test_a_changed_field_fails_beside_a_changed_work_counter(
    graph: ReferenceGraph, exports: Path, tmp_path: Path
) -> None:
    def tamper(
        name: str, params: dict[str, Any], rows: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        rows = _steps(name, params, rows)
        if name == READ_MONEY_FLOW and rows[0].get("status") == "ok":
            for vertex in _blocks(rows, "accounts"):
                vertex["attributes"]["in_amount_90d"] += 1
        return rows

    found = run(Live(graph, tamper), CONFIG, exports=exports, record=tmp_path / "parity.toml")
    assert found.failed == (READ_MONEY_FLOW,)
    what = found.queries[READ_MONEY_FLOW].mismatches[0].what
    assert "in_amount_90d" in what and "scan_steps" not in what


def test_only_the_work_counters_of_b1_are_left_out_of_what_parity_compares(
    graph: ReferenceGraph,
) -> None:
    assert WORK_COUNTERS == {READ_MONEY_FLOW: frozenset({"scan_steps"})}
    accounts = ["a_collector", "a_mule"]
    answer = read_money_flow.parse(
        graph.run(READ_MONEY_FLOW, read_money_flow.params(T, accounts)), accounts
    )
    assert answer.accounts and answer.tracked_inflows

    def changed(value: object) -> object:
        if isinstance(value, bool):
            return not value
        if isinstance(value, int):
            return value + 1
        if isinstance(value, float):
            return value + 0.5
        return f"{value}x"

    def compared(other: MoneyFlows) -> bool:
        """Whether parity sees a difference between the answer and other."""
        return bool(
            differences(_canonical(MONEY_FLOW, answer), _canonical(MONEY_FLOW, other), TOLERANCE)
        )

    equal: set[str] = set()
    for key, rows in (("accounts", answer.accounts), ("tracked_inflows", answer.tracked_inflows)):
        for field in fields(rows[0]):
            row = replace(rows[0], **{field.name: changed(getattr(rows[0], field.name))})
            if not compared(replace(answer, **{key: (row, *rows[1:])})):
                equal.add(field.name)
    assert equal == {"scan_steps"}


def test_a_live_answer_the_parser_refuses_fails_its_query(
    graph: ReferenceGraph, exports: Path, tmp_path: Path
) -> None:
    def tamper(
        name: str, params: dict[str, Any], rows: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        # B1 stops with a status when a heap did not iterate in order.
        if name == READ_MONEY_FLOW and rows[0].get("status") == "ok":
            return [{"status": "unsorted_heap", "accounts": []}]
        # B4 gives a subgraph a payment its list of payments lacks.
        if name == READ_PAYMENT_EVENTS and rows[0].get("status") == "ok":
            for block in rows:
                if "payments" in block:
                    block["payments"].pop()
        return rows

    found = run(Live(graph, tamper), CONFIG, exports=exports, record=tmp_path / "parity.toml")
    assert found.failed == (READ_PAYMENT_EVENTS, READ_MONEY_FLOW) or found.failed == (
        READ_MONEY_FLOW,
        READ_PAYMENT_EVENTS,
    )
    assert "unsorted_heap" in found.queries[READ_MONEY_FLOW].mismatches[0].what
    assert "ValueError" in found.queries[READ_PAYMENT_EVENTS].mismatches[0].what


def test_a_live_request_that_raises_is_a_mismatch_not_a_crash(
    graph: ReferenceGraph, exports: Path, tmp_path: Path
) -> None:
    def tamper(
        name: str, params: dict[str, Any], rows: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if name == READ_ACTIVITY_CHANGE and "account_ids" in params and not params["account_ids"]:
            raise RuntimeError("Parameter account_ids must not be empty")
        return rows

    found = run(Live(graph, tamper), CONFIG, exports=exports, record=tmp_path / "parity.toml")
    assert found.failed == (READ_ACTIVITY_CHANGE,)
    mismatch = found.queries[READ_ACTIVITY_CHANGE].mismatches[0]
    assert "no account" in mismatch.where and "RuntimeError" in mismatch.what


def test_a_refusal_the_live_graph_words_otherwise_fails(
    graph: ReferenceGraph, exports: Path, tmp_path: Path
) -> None:
    def tamper(
        name: str, params: dict[str, Any], rows: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if name == READ_COUNTERPARTIES and rows == [{"status": "unknown_account"}]:
            return [{"status": "invisible_account"}]
        return rows

    found = run(Live(graph, tamper), CONFIG, exports=exports, record=tmp_path / "parity.toml")
    assert found.failed == (READ_COUNTERPARTIES,)
    mismatch = found.queries[READ_COUNTERPARTIES].mismatches[0]
    assert "an unknown account" in mismatch.where and "invisible_account" in mismatch.what


def _late_world() -> WorldBuilder:
    """The batch world and one customer who joins two days after the first scored Monday: on that
    Monday an account that does not exist yet, which the refusal probes of an account and of a
    hub that do not exist at t are sent for (the module's other worlds have none)."""
    w = fixtures.batch_world()
    w.customer("late", T + 2 * DAY_MS)
    return w


PER_ACCOUNT_QUERIES = (
    READ_MONEY_FLOW,
    READ_COUNTERPARTIES,
    READ_ACTIVITY_CHANGE,
    READ_PAYMENT_EVENTS,
    READ_SHARED_IDENTITY,
    READ_COMMUNITY,
)
TAKES_HUBS_QUERIES = (
    READ_COUNTERPARTIES,
    READ_PAYMENT_EVENTS,
    READ_SHARED_IDENTITY,
    READ_COMMUNITY,
)


def test_an_account_and_a_hub_that_do_not_exist_at_t_are_refused_as_such(tmp_path: Path) -> None:
    w = _late_world()
    write_export(tmp_path / "exports" / "e", w.tables())
    refused: dict[tuple[str, str], int] = {}

    def record(
        name: str, params: dict[str, Any], rows: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        status = rows[0].get("status", "")
        if status in ("invisible_account", "invisible_hub"):
            refused[(name, status)] = refused.get((name, status), 0) + 1
        return rows

    found = run(
        Live(w.graph(), record), CONFIG, exports=tmp_path / "exports", record=tmp_path / "p.toml"
    )
    assert found.failed == ()
    # On the first Monday, the one with an account that does not exist yet, every per-account
    # query is sent that account, and each that takes the hubs is sent it as a hub too.
    assert refused == {(name, "invisible_account"): 1 for name in PER_ACCOUNT_QUERIES} | {
        (name, "invisible_hub"): 1 for name in TAKES_HUBS_QUERIES
    }
    # The probes are checks of their query: a request more for B1, two more for B2.
    assert found.queries[READ_MONEY_FLOW].checks == 2 * (3 + 2 + 4) + 1
    assert found.queries[READ_COUNTERPARTIES].checks == 2 * (3 + 2 + 4) + 2


def test_a_late_account_or_hub_refused_in_other_words_fails(tmp_path: Path) -> None:
    w = _late_world()
    write_export(tmp_path / "exports" / "e", w.tables())

    def tamper(
        name: str, params: dict[str, Any], rows: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if name == READ_MONEY_FLOW and rows == [{"status": "invisible_account"}]:
            return [{"status": "unknown_account"}]
        if name == READ_COMMUNITY and rows == [{"status": "invisible_hub"}]:
            return [{"status": "invisible_account"}]
        return rows

    found = run(
        Live(w.graph(), tamper), CONFIG, exports=tmp_path / "exports", record=tmp_path / "p.toml"
    )
    assert set(found.failed) == {READ_MONEY_FLOW, READ_COMMUNITY}
    account = found.queries[READ_MONEY_FLOW].mismatches
    assert len(account) == 1 and "an account that does not exist at t" in account[0].where
    assert "unknown_account" in account[0].what and "invisible_account" in account[0].what
    hub = found.queries[READ_COMMUNITY].mismatches
    assert len(hub) == 1 and "a hub that does not exist at t" in hub[0].where
    assert "invisible_hub" in hub[0].what


def test_an_account_alone_must_equal_itself_among_the_others(
    graph: ReferenceGraph, exports: Path, tmp_path: Path
) -> None:
    def tamper(
        name: str, params: dict[str, Any], rows: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        # Every account's cents received in the 90 days grow with the size of its request:
        # per-request state shared between the accounts of a request.
        if name == READ_MONEY_FLOW and len(params.get("account_ids", [])) > 1:
            if rows[0].get("status") == "ok":
                for vertex in _blocks(rows, "accounts"):
                    vertex["attributes"]["in_amount_90d"] += len(params["account_ids"])
        return rows

    found = run(Live(graph, tamper), CONFIG, exports=exports, record=tmp_path / "parity.toml")
    assert found.failed == (READ_MONEY_FLOW,)
    where = [item.where for item in found.queries[READ_MONEY_FLOW].mismatches]
    # Against the reference, in every request of 8; and against itself, alone and among 7
    # others (a request of 8 holds the account and 7 others).
    assert any("8 accounts from" in place for place in where)
    assert any("alone against 7 others" in place for place in where)


def test_an_account_alone_is_asked_of_the_live_graph_only(
    tables: dict[str, pd.DataFrame], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The reference graph has no use for the alone answer, which is compared with the live
    # answer among the others: no request of one account is sent to it, in 2 Mondays x 2
    # accounts x 6 queries.
    reference = ReferenceGraph(tables)
    asked: list[tuple[str, int]] = []
    original = reference.run

    def counting(name: str, params: dict[str, Any], **keywords: Any) -> list[dict[str, Any]]:
        asked.append((name, len(params.get("account_ids", []))))
        return original(name, params, **keywords)

    monkeypatch.setattr(reference, "run", counting)
    live = ParityExecutor(Live(ReferenceGraph(tables)))
    queries, _ = run_on(live, reference, (T, T + W), CONFIG.parity)
    assert all(result.passed for result in queries.values())
    assert any(size == 8 for _, size in asked) and not [name for name, size in asked if size == 1]
    # The alone answers were still compared, with the live answers among the others: on each
    # Monday 3 requests, 2 accounts alone and 4 refusals.
    assert queries[READ_MONEY_FLOW].checks == 2 * (3 + 2 + 4)


def _crowd_world() -> WorldBuilder:
    """The reference's own test world (test_reference_community): p_s and p_m, a member by a
    payment, are enrolled on d_crowd with 6 others, and p_u, who never enrolled, sent a payment
    seen on it, a ninth customer: a crowd read_hubs lists."""
    w = WorldBuilder()
    w.external("x_ext", fixtures.OPENED)
    for name in ("s", "m", *(f"k{n}" for n in range(1, 7))):
        w.customer(name, fixtures.JOINED)
        w.device(f"p_{name}", "d_crowd", fixtures.JOINED)
    w.pay("z_m_s", "a_m", "a_s", T - 5 * DAY_MS, 50.0, "zelle")
    w.customer("u", fixtures.JOINED)
    w.pay("x_u_seen", "a_u", "x_ext", T - 20 * DAY_MS, 5.0, "ach", device="d_crowd")
    return w


CROWD_CONFIG = replace(
    CONFIG,
    calendar=CalendarConfig(history_days=182, scored_mondays=1),
    parity=ParityConfig(mondays=1, accounts=9, together=4, alone=2),
)


def test_a_world_with_a_crowd_between_two_members_is_counted(tmp_path: Path) -> None:
    # Every one of the 9 scorable accounts holds d_crowd as its customer's current link, or on
    # a payment of the last 30 days: 9 communities touch a leaf, and two hold a crowd between
    # two members, a_s's and a_m's, whose customers hold it both.
    w = _crowd_world()
    write_export(tmp_path / "exports" / "e", w.tables())
    found = run(
        Live(w.graph()), CROWD_CONFIG, exports=tmp_path / "exports", record=tmp_path / "p.toml"
    )
    assert found.failed == ()
    assert found.communities == Communities(9, 9, 2)
    assert not any("met no crowd" in line for line in found.lines())


def test_b5s_leaf_flags_are_read_against_the_live_read_hubs(tmp_path: Path) -> None:
    w = _crowd_world()
    write_export(tmp_path / "exports" / "e", w.tables())

    def tamper(
        name: str, params: dict[str, Any], rows: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        # The live read_hubs forgets the crowd, which B5 still flags a leaf.
        if name == READ_HUBS and rows[0].get("status") == "ok":
            for block in rows:
                if "devices" in block:
                    block["devices"] = []
        return rows

    found = run(
        Live(w.graph(), tamper),
        CROWD_CONFIG,
        exports=tmp_path / "exports",
        record=tmp_path / "p.toml",
    )
    assert READ_SHARED_IDENTITY in found.failed and READ_HUBS in found.failed
    where = [item.where for item in found.queries[READ_SHARED_IDENTITY].mismatches]
    assert any("leaf flags against read_hubs" in place for place in where)
    what = next(
        item.what
        for item in found.queries[READ_SHARED_IDENTITY].mismatches
        if "leaf flags" in item.where
    )
    assert "device d_crowd" in what and "is flagged a leaf" in what


def test_the_export_is_the_folder_whose_counts_and_identity_equal_the_live_worlds(
    graph: ReferenceGraph, tables: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    live = world_of(graph)
    other = fixtures.batch_world()
    other.pay("x_extra", "a_h1", "a_h2", T - DAY_MS, 1.0, "ach")
    write_export(tmp_path / "a_other_world", other.tables())
    write_export(tmp_path / "b_the_world", tables)
    folder, found = find_export(tmp_path, live)
    assert folder == tmp_path / "b_the_world"
    assert world_of(found) == live
    # The root may itself be the export.
    assert find_export(tmp_path / "b_the_world", live)[0] == tmp_path / "b_the_world"


def test_no_export_is_refused_naming_why(
    graph: ReferenceGraph, tables: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    live = world_of(graph)
    with pytest.raises(ParityError, match="is no folder"):
        find_export(tmp_path / "missing", live)
    with pytest.raises(ParityError, match="No export folder with a export_manifest.json"):
        find_export(tmp_path, live)
    # An export with other counts per table.
    other = fixtures.batch_world()
    other.pay("x_extra", "a_h1", "a_h2", T - DAY_MS, 1.0, "ach")
    write_export(tmp_path / "counts", other.tables())
    with pytest.raises(ParityError, match="has the loaded world's counts per table"):
        find_export(tmp_path, live)
    # An export with the same counts and another identity: one link's end moved to another
    # vertex of another degree, which no count shows but the degree sums do.
    moved = {name: frame.copy() for name, frame in tables.items()}
    links = moved["Party_Uses_Device"]
    other_device = next(
        i for i in range(1, len(links)) if links.loc[i, "to_id"] != links.loc[0, "to_id"]
    )
    links.loc[0, "to_id"] = links.loc[other_device, "to_id"]
    write_export(tmp_path / "degrees", moved)
    with pytest.raises(ParityError, match=r"nearest differs in .*Party_Uses_Device"):
        find_export(tmp_path, live)


def test_the_samples_are_seeded_and_oversample_the_true_mules(graph: ReferenceGraph) -> None:
    config = CONFIG.parity
    mondays = DEFAULT_CONFIG.calendar.mondays()
    first = sample_mondays(mondays, config)
    assert first == sample_mondays(mondays, config) and len(first) == 2
    assert first == tuple(sorted(first)) and set(first) <= set(mondays)
    assert sample_mondays(mondays, replace(config, seed=1)) != first
    assert sample_mondays(mondays[:2], config) == mondays[:2]
    mules, others = sample_accounts(graph, T, config)
    # A quarter of 24 accounts are true mules, the rest are not.
    assert len(mules) == 6 and len(others) == 18
    assert set(mules) <= MULES and set(others).isdisjoint(MULES)
    assert (mules, others) == sample_accounts(graph, T, config)
    assert sample_accounts(graph, T, replace(config, seed=3))[1] != others
    # With fewer mules than a quarter, all of them, and the rest ordinary accounts.
    few, rest = sample_accounts(graph, T, replace(config, accounts=8, mule_share=0.5))
    assert len(few) == 4 and len(rest) == 4
    everyone, nobody = sample_accounts(graph, T, replace(config, accounts=400, mule_share=0.9))
    assert set(everyone) == MULES and len(nobody) == 224 - len(MULES)


def test_two_requests_that_fail_alike_prove_nothing(graph: ReferenceGraph) -> None:
    comparison = _Comparison(ParityExecutor(Live(graph)), graph, CONFIG.parity)
    result = comparison.found[READ_HUBS]
    comparison.check(READ_HUBS, "both fail", Failure("refused"), Failure("refused"))
    comparison.check(READ_HUBS, "one fails", {"a": [1]}, Failure("refused"))
    comparison.check(READ_HUBS, "equal", {"a": [1]}, {"a": [1]})
    assert (result.checks, result.mismatched) == (3, 2)
    assert [item.what for item in result.mismatches] == [
        "live: refused; reference: refused",
        "live: answered; reference: refused",
    ]


def test_two_answers_differ_where_they_differ() -> None:
    assert differences({"a": [1, 2.5, "x"]}, {"a": [1, 2.5, "x"]}, 1e-12) == []
    # A double is the same within the relative tolerance, whatever the integer or float of it.
    assert differences(0.3, 0.30000000000000004, 1e-12) == []
    assert differences(3, 3.0, 1e-12) == []
    assert differences(0.3, 0.3000001, 1e-12) == ["answer is 0.3 live, 0.3000001 in the reference"]
    # Integers, strings and booleans are exact, and a boolean is no integer.
    assert differences(7, 8, 0.5) == ["answer is 7 live, 8 in the reference"]
    assert differences(True, 1, 1e-12) == ["answer is True live, 1 in the reference"]
    assert differences("a", "b", 1e-12) == ["answer is 'a' live, 'b' in the reference"]
    assert differences({"a": 1}, {"b": 1}, 1e-12) == [
        "answer.a only in live",
        "answer.b only in reference",
    ]
    found = differences({"rows": [{"x": 1}, {"x": 2}]}, {"rows": [{"x": 1}]}, 1e-12)
    assert found == ["answer.rows has 2 rows live, 1 in the reference"]
    # At most four places are named.
    many = differences(list(range(9)), list(range(1, 10)), 1e-12)
    assert len(many) == 4 and many[0] == "answer[0] is 0 live, 1 in the reference"


@pytest.mark.graph
def test_the_loaded_world_matches_its_export(tmp_path: Path) -> None:
    """mule parity on the live MulePatternLearner workspace, which needs a world loaded by the
    loader and its export under MULE_EXPORT_DIR (the 5,000-person test)."""
    from mule_pattern_learner.paths import export_root
    from mule_pattern_learner.tigergraph.connection import ConnectionSettings
    from mule_pattern_learner.tigergraph.executor import ParityExecutor, TigerGraphExecutor
    from mule_pattern_learner.tigergraph.queries import read_world

    executor = TigerGraphExecutor(settings=ConnectionSettings())
    found = read_world.parse(ParityExecutor(executor).run(READ_WORLD, read_world.params()))
    if found.vertices["Account"] == 0:
        pytest.skip("the graph holds no world: load one with `mule-load push` first")
    try:
        root = export_root()
    except ValueError as error:
        pytest.skip(f"no export folder is set: {error}")
    parity = run(executor, DEFAULT_CONFIG, exports=root, record=tmp_path / "parity.toml")
    assert parity.failed == (), "\n".join(parity.lines())


def test_a_config_names_the_calendar_the_sample_is_drawn_from() -> None:
    assert isinstance(CONFIG, RunConfig) and CONFIG.calendar.mondays()[:2] == (T, T + W)
