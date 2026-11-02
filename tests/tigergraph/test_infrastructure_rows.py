"""The parameter builders and row parsers of the six infrastructure queries and of mark_crowds,
on answers written by hand in TigerGraph's shape: a status row first, then one row per PRINT, a printed vertex set
as a list of vertices with their projected fields under "attributes".

Each parser refuses a status row, an answer of another version, a page out of id order, and the
values the contract does not allow; the answer key's parser is imported by evaluation and
testing alone.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import mule_pattern_learner
from mule_pattern_learner.tigergraph.queries import (
    LIST_ACCOUNTS,
    MARK_CROWDS,
    PROFILE_HUBS,
    READ_ANSWER_KEY,
    READ_HUBS,
    READ_LABELS,
    READ_WORLD,
    list_accounts,
    mark_crowds,
    profile_hubs,
    read_hubs,
    read_labels,
    read_world,
)
from mule_pattern_learner.tigergraph.queries import read_answer_key as answer_key
from mule_pattern_learner.tigergraph.queries.read_world import (
    EDGE_TYPES,
    SECONDS,
    VERTEX_TYPES,
)
from mule_pattern_learner.tigergraph.queries.rows import (
    DAY_MS,
    LOOKBACK_MS,
    PAGE_LIMIT,
    QueryRefused,
)

T_MS = 1_711_929_600_000  # 2024-04-01 00:00 UTC, a Monday


def _head(name: str, **fields: object) -> dict[str, object]:
    del name
    return {"status": "ok", **fields}


def _vertices(kind: str, rows: list[dict[str, Any]], id_field: str) -> list[dict[str, Any]]:
    """A printed projection of a vertex set, as TigerGraph prints it."""
    return [{"v_id": row[id_field], "v_type": kind, "attributes": row} for row in rows]


# The builders.


def test_a_query_with_t_takes_a_whole_second_after_the_lookback() -> None:
    for build in (read_hubs.params, mark_crowds.params, profile_hubs.params):
        assert build(T_MS) == {"t_ms": T_MS}
    assert list_accounts.params(T_MS, "", 500) == {"t_ms": T_MS, "after_id": "", "page": 500}
    # Any whole second is a time, mid-week and mid-day as well as a Monday at midnight: a
    # Tuesday at 13:47:21, a Sunday a second before midnight.
    for t_ms in (T_MS + DAY_MS + 49_641_000, T_MS + 7 * DAY_MS - 1_000, T_MS + 1_000):
        assert read_hubs.params(t_ms) == {"t_ms": t_ms}
        assert profile_hubs.params(t_ms) == {"t_ms": t_ms}
        assert list_accounts.params(t_ms, "", 1)["t_ms"] == t_ms
    # A millisecond that is no whole second, in the middle of a second or a millisecond late
    # on a Monday, and a boolean are refused.
    for t_ms in (T_MS + 1, T_MS + DAY_MS + 500, T_MS - 999, True):
        with pytest.raises(ValueError, match="whole second"):
            read_hubs.params(t_ms)  # pyright: ignore[reportArgumentType]
        with pytest.raises(ValueError, match="whole second"):
            mark_crowds.params(t_ms)  # pyright: ignore[reportArgumentType]
        with pytest.raises(ValueError, match="whole second"):
            list_accounts.params(t_ms, "", 1)  # pyright: ignore[reportArgumentType]
    # The lookback is 90 days: t at the lookback's length or inside it is too early, a second
    # after it the earliest t.
    assert LOOKBACK_MS == 90 * DAY_MS
    assert read_world.params() == {}
    assert profile_hubs.params(LOOKBACK_MS + 1_000) == {"t_ms": LOOKBACK_MS + 1_000}
    for t_ms in (0, 88 * DAY_MS, LOOKBACK_MS - 1_000, LOOKBACK_MS):
        with pytest.raises(ValueError, match="whole second"):
            profile_hubs.params(t_ms)


def _page_of_accounts(after_id: str, page: int) -> dict[str, Any]:
    return list_accounts.params(T_MS, after_id, page)


@pytest.mark.parametrize("build", [_page_of_accounts, read_labels.params, answer_key.params])
def test_a_page_holds_1_to_10000_accounts_after_an_id(build: Callable[..., dict[str, Any]]) -> None:
    assert build("a_09", 1)["after_id"] == "a_09" and build("", PAGE_LIMIT)["page"] == PAGE_LIMIT
    for page in (0, PAGE_LIMIT + 1, 2.0, True):
        with pytest.raises(ValueError, match="page"):
            build("", page)
    with pytest.raises(ValueError, match="after_id"):
        build(None, 10)


# The paged queries.


ACCOUNTS = [{"account_id": "a_01", "owner_id": "p_01"}, {"account_id": "a_02", "owner_id": "p_01"}]


def _accounts(rows: list[dict[str, Any]], name: str = LIST_ACCOUNTS) -> list[object]:
    return [_head(name), {"accounts": _vertices("Account", rows, "account_id")}]


def test_a_page_of_scorable_accounts_parses_in_id_order() -> None:
    page = list_accounts.parse(_accounts(ACCOUNTS), "", 10)
    assert [(a.account_id, a.owner_id) for a in page] == [("a_01", "p_01"), ("a_02", "p_01")]
    assert list_accounts.parse(_accounts([]), "a_02", 10) == ()


@pytest.mark.parametrize(
    ("rows", "after_id", "page", "message"),
    [
        (ACCOUNTS[::-1], "", 10, "strictly increasing"),
        ([ACCOUNTS[0], ACCOUNTS[0]], "", 10, "strictly increasing"),
        (ACCOUNTS, "a_01", 10, "strictly increasing"),
        (ACCOUNTS, "", 1, "to a page of 1"),
        ([{"account_id": "a_01", "owner_id": ""}], "", 10, "without an owner"),
        ([{"account_id": "a_01", "owner_id": "p_01", "opened_ms": 5}], "", 10, "fields"),
    ],
)
def test_a_page_out_of_order_too_long_or_unowned_is_refused(
    rows: list[dict[str, Any]], after_id: str, page: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        list_accounts.parse(_accounts(rows), after_id, page)


PARSERS: list[tuple[str, Callable[[list[object]], object]]] = [
    (LIST_ACCOUNTS, lambda rows: list_accounts.parse(rows, "", 10)),
    (READ_LABELS, lambda rows: read_labels.parse(rows, "", 10)),
    (READ_ANSWER_KEY, lambda rows: answer_key.parse(rows, "", 10)),
    (READ_HUBS, read_hubs.parse),
    (MARK_CROWDS, mark_crowds.parse),
    (READ_WORLD, read_world.parse),
    (PROFILE_HUBS, profile_hubs.parse),
]


@pytest.mark.parametrize(("name", "parse"), PARSERS)
def test_every_parser_refuses_a_status_row(
    name: str, parse: Callable[[list[object]], object]
) -> None:
    with pytest.raises(QueryRefused) as refused:
        parse([{"status": "invalid_parameters"}])
    assert (refused.value.query, refused.value.status) == (name, "invalid_parameters")


def test_a_printed_set_needs_its_vertices_attributes() -> None:
    with pytest.raises(ValueError, match="not a vertex's attributes"):
        list_accounts.parse([_head(LIST_ACCOUNTS), {"accounts": ACCOUNTS}], "", 10)
    with pytest.raises(ValueError, match="printed accounts 0 times"):
        list_accounts.parse([_head(LIST_ACCOUNTS)], "", 10)
    with pytest.raises(ValueError, match="not a list of vertices"):
        list_accounts.parse([_head(LIST_ACCOUNTS), {"accounts": {}}], "", 10)


MULES = [
    {"account_id": "a_03", "mule_onset_ms": 1_000, "mule_last_ms": 1_000},
    {"account_id": "a_07", "mule_onset_ms": 1_000, "mule_last_ms": 9_000},
]


def test_identified_mules_parse_with_their_activity_period() -> None:
    mules = read_labels.parse(_accounts(MULES, READ_LABELS), "", 10)
    assert [(m.account_id, m.mule_onset_ms, m.mule_last_ms) for m in mules] == [
        ("a_03", 1_000, 1_000),
        ("a_07", 1_000, 9_000),
    ]


@pytest.mark.parametrize(
    ("onset", "last"),
    [(0, 0), (0, 5), (9_001, 9_000)],
    ids=["none", "no onset", "onset after last"],
)
def test_an_identified_mule_without_0_below_onset_up_to_last_is_refused(
    onset: int, last: int
) -> None:
    rows = [MULES[0], {"account_id": "a_07", "mule_onset_ms": onset, "mule_last_ms": last}]
    with pytest.raises(ValueError, match="not 0 < onset <= last"):
        read_labels.parse(_accounts(rows, READ_LABELS), "", 10)


def test_the_querys_own_refusal_of_an_activity_period_names_the_accounts() -> None:
    refusal = {"status": "invalid_activity", "accounts": ["a_07"]}
    with pytest.raises(QueryRefused) as refused:
        read_labels.parse([refusal], "", 10)
    assert refused.value.row == refusal


TRUTH: dict[str, Any] = {
    "account_id": "a_01",
    "is_mule": 1,
    "mule_ring_id": 4,
    "mule_onset_ms": 1_000,
    "mule_last_ms": 9_000,
    "opened_ms": 500,
    "account_type": "deposit",
    "owned": True,
}


def test_the_answer_key_parses_every_internal_accounts_truth() -> None:
    plain = {**TRUTH, "account_id": "a_02", "is_mule": 0, "mule_ring_id": -1, "owned": False}
    plain |= {"mule_onset_ms": 0, "mule_last_ms": 0, "account_type": "gl"}
    truths = answer_key.parse(_accounts([TRUTH, plain], READ_ANSWER_KEY), "", 10)
    assert [(t.account_id, t.is_mule, t.mule_ring_id, t.owned) for t in truths] == [
        ("a_01", 1, 4, True),
        ("a_02", 0, -1, False),
    ]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"account_type": "savings"}, "account_type 'savings'"),
        ({"is_mule": -1}, "is_mule"),
        ({"is_mule": 2}, "is_mule 2"),
        ({"mule_ring_id": -2}, "ring -2"),
        ({"opened_ms": -1}, "opened_ms"),
        ({"owned": 1}, "owned"),
    ],
)
def test_the_answer_key_refuses_values_outside_the_contract(
    change: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        answer_key.parse(_accounts([{**TRUTH, **change}], READ_ANSWER_KEY), "", 10)


def test_only_evaluation_and_testing_import_the_answer_keys_parser() -> None:
    # The import contract in pyproject.toml covers every other package; inside tigergraph an
    # import is not the contract's to see, so this reads every module's imports.
    package = Path(mule_pattern_learner.__file__).parent
    importers: list[str] = []
    for path in sorted(package.rglob("*.py")):
        module = path.relative_to(package).as_posix()
        for node in ast.walk(ast.parse(path.read_text())):
            names: list[str] = []
            if isinstance(node, ast.ImportFrom):
                names = [node.module or "", *(alias.name for alias in node.names)]
            elif isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            if any(name.rsplit(".", 1)[-1] == "read_answer_key" for name in names):
                importers.append(module)
    assert all(module.startswith(("evaluation/", "testing/")) for module in importers), importers


# The hubs.


def _leaf(node_id: str, customers: int, observed_only: bool = False) -> dict[str, Any]:
    return {
        "node_id": node_id,
        "customers_90d": customers,
        "observed_only": observed_only,
        "hub": customers > 200,
    }


def _hubs(
    accounts: list[dict[str, Any]], devices: list[dict[str, Any]], ips: list[dict[str, Any]]
) -> list[object]:
    return [
        _head(READ_HUBS),
        {"accounts": _vertices("Account", accounts, "account_id")},
        {"devices": _vertices("Device", devices, "node_id")},
        {"ips": _vertices("IP", ips, "node_id")},
    ]


def test_the_hubs_parse_sorted_by_id_with_their_measures() -> None:
    hubs = read_hubs.parse(
        _hubs(
            [
                {"account_id": "ledger", "customers_90d": 5_000},
                {"account_id": "biller", "customers_90d": 201},
            ],
            [_leaf("d_2", 9), _leaf("d_1", 12, observed_only=True)],
            [_leaf("ip_1", 201)],
        )
    )
    assert [(a.account_id, a.customers_90d) for a in hubs.accounts] == [
        ("biller", 201),
        ("ledger", 5_000),
    ]
    assert [(d.node_id, d.observed_only, d.hub) for d in hubs.devices] == [
        ("d_1", True, False),
        ("d_2", False, False),
    ]
    assert [(i.node_id, i.customers_90d, i.hub) for i in hubs.ips] == [("ip_1", 201, True)]


@pytest.mark.parametrize(
    ("accounts", "devices", "ips", "message"),
    [
        ([{"account_id": "a", "customers_90d": 200}], [], [], "with 200 customers"),
        ([], [_leaf("d_1", 8)], [], "among its devices"),
        ([], [], [{**_leaf("ip_1", 201), "hub": False}], "among its ips"),
        ([], [], [{**_leaf("ip_1", 200), "hub": True}], "among its ips"),
        ([], [_leaf("d_1", 9), _leaf("d_1", 10)], [], "repeated id in devices"),
    ],
)
def test_a_hub_or_crowd_below_its_threshold_or_repeated_is_refused(
    accounts: list[dict[str, Any]],
    devices: list[dict[str, Any]],
    ips: list[dict[str, Any]],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        read_hubs.parse(_hubs(accounts, devices, ips))


# The crowds.


def _crowds(devices: list[dict[str, Any]], ips: list[dict[str, Any]]) -> list[object]:
    return [
        _head(MARK_CROWDS),
        {"devices": _vertices("Device", devices, "node_id")},
        {"ips": _vertices("IP", ips, "node_id")},
    ]


def test_the_crowds_parse_sorted_by_id_with_their_measures() -> None:
    crowds = mark_crowds.parse(
        _crowds([_leaf("d_2", 9), _leaf("d_1", 12, observed_only=True)], [_leaf("ip_1", 201)])
    )
    assert [(d.node_id, d.customers_90d, d.observed_only, d.hub) for d in crowds.devices] == [
        ("d_1", 12, True, False),
        ("d_2", 9, False, False),
    ]
    assert [(i.node_id, i.customers_90d, i.hub) for i in crowds.ips] == [("ip_1", 201, True)]
    assert mark_crowds.parse(_crowds([], [])) == mark_crowds.Crowds((), ())


@pytest.mark.parametrize(
    ("devices", "ips", "message"),
    [
        ([_leaf("d_1", 8)], [], "among its devices"),
        ([], [{**_leaf("ip_1", 201), "hub": False}], "among its ips"),
        ([], [{**_leaf("ip_1", 200), "hub": True}], "among its ips"),
        ([_leaf("d_1", 9), _leaf("d_1", 10)], [], "repeated id in devices"),
        ([{**_leaf("d_1", 9), "crowd_at_ms": T_MS}], [], "has the fields"),
    ],
)
def test_a_crowd_below_its_threshold_with_a_wrong_hub_flag_or_repeated_is_refused(
    devices: list[dict[str, Any]], ips: list[dict[str, Any]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        mark_crowds.parse(_crowds(devices, ips))


HISTOGRAMS: dict[str, dict[str, int]] = {
    "accounts": {"0": 900, "3": 40, "250": 1},
    "registered_devices": {"1": 30, "2": 5},
    "observed_devices": {"10": 1},
    "registered_ips": {"90": 2},
    "observed_ips": {},
}


def test_the_hub_profile_parses_each_histogram_by_measure() -> None:
    profile = profile_hubs.parse([_head(PROFILE_HUBS, **HISTOGRAMS)])
    assert profile.accounts == {0: 900, 3: 40, 250: 1}
    assert list(profile.accounts) == [0, 3, 250]
    assert profile.registered_devices == {1: 30, 2: 5} and profile.observed_ips == {}


@pytest.mark.parametrize(
    ("name", "histogram", "message"),
    [
        ("accounts", {"x": 1}, "measure 'x'"),
        ("accounts", {"-1": 1}, "measure '-1'"),
        ("accounts", {"3": -1}, "not counts by name"),
        ("accounts", {"3": 0}, "0 at the measure"),
        ("registered_devices", {"0": 4}, "measure '0'"),
        ("observed_ips", [1, 2], "not counts by name"),
    ],
)
def test_a_histogram_that_is_no_count_by_measure_is_refused(
    name: str, histogram: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        profile_hubs.parse([_head(PROFILE_HUBS, **{**HISTOGRAMS, name: histogram})])
    fewer = {key: value for key, value in HISTOGRAMS.items() if key != name}
    with pytest.raises(ValueError, match="printed the fields"):
        profile_hubs.parse([_head(PROFILE_HUBS, **fewer)])


# The world.


def _world(**changes: object) -> dict[str, object]:
    """read_world's answer for a small world, as TigerGraph prints it: no key for a type or
    value with no row."""
    fields: dict[str, object] = {
        "vertices": {"Party": 3, "Account": 4, "Payment_Transaction": 5, "Device": 1},
        "edges": {"Party_Owns_Account": 4, "Transaction_From_Account": 5},
        "account_types": {"deposit": 3, "gl": 1},
        "payment_rails": {"ach": 4, "unknown": 1},
        "external_accounts": 1,
        "seconds": {"Party.joined_ms": 3_000, "Payment_Transaction.event_ts_ms": 9_000_000},
        "cents": {"Payment_Transaction": 12_345},
        "squared_degrees": {"Party_Owns_Account": 6, "Transaction_From_Account": 5},
        "squared_target_degrees": {"Party_Owns_Account": 4, "Transaction_From_Account": 9},
        "degree_products": {"Party_Owns_Account": 6, "Transaction_From_Account": 9},
        "first_joined_ms": 1_000,
        "first_event_ms": 1_704_067_200_000,
        "last_event_ms": 1_743_292_800_000,
    }
    return _head(READ_WORLD, **{**fields, **changes})


def test_the_world_parses_with_every_contract_key_and_zero_for_those_with_no_row() -> None:
    world = read_world.parse([_world()])
    assert list(world.vertices) == list(VERTEX_TYPES) and len(world.edges) == len(EDGE_TYPES) == 15
    assert world.vertices["Account"] == 4 and world.vertices["Token"] == 0
    assert world.account_types == {
        "deposit": 3,
        "credit": 0,
        "brokerage": 0,
        "gl": 1,
        "unknown": 0,
    }
    assert world.payment_rails["unknown"] == 1 and world.payment_rails["cash"] == 0
    assert list(world.seconds) == list(SECONDS) and world.seconds["Account.closed_ms"] == 0
    assert world.cents == {"Payment_Transaction": 12_345, "Zelle_Transfer": 0}
    assert world.squared_target_degrees["Transaction_From_Account"] == 9
    assert (world.external_accounts, world.first_joined_ms) == (1, 1_000)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"account_types": {"deposit": 3, "savings": 1}}, "account_types of \\['savings'\\]"),
        ({"payment_rails": {"zelle": 2}}, "payment_rails of \\['zelle'\\]"),
        ({"vertices": {"Address": 2}}, "vertices of \\['Address'\\]"),
        ({"edges": {"Account_Owned_By_Party": 4}}, "edges of \\['Account_Owned_By_Party'\\]"),
        ({"seconds": {"Account.mule_onset_ms": 5}}, "seconds of"),
        ({"cents": {"Payment_Transaction": -1}}, "not counts by name"),
        ({"external_accounts": -1}, "external_accounts -1"),
        ({"last_event_ms": True}, "last_event_ms True"),
        ({"first_event_ms": 1.5}, "first_event_ms 1.5"),
        ({"squared_degrees": [4]}, "not counts by name"),
    ],
)
def test_the_world_refuses_values_outside_the_contract(
    change: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        read_world.parse([_world(**change)])


def test_the_world_refuses_a_missing_or_an_extra_field() -> None:
    head = _world()
    del head["cents"]
    with pytest.raises(ValueError, match="printed the fields"):
        read_world.parse([head])
    with pytest.raises(ValueError, match="printed the fields"):
        read_world.parse([_world(reverse_edges={})])
