"""The reference graph's core, its six infrastructure queries and mark_crowds, on small worlds
whose answers are worked out by hand from docs/design.md ("Time", "The scored population", "Shared rules"
and the sections of each query).

Every answer goes through ReferenceGraph.run and the query's own parser, so it has the shape
TigerGraph prints and the parser's checks hold of it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, replace

import pandas as pd
import pytest

from mule_pattern_learner.testing.reference import ReferenceGraph
from mule_pattern_learner.testing.reference.answer import valid_time
from mule_pattern_learner.testing.reference.graph import DEVICE, IP, Link, cents, frame
from mule_pattern_learner.testing.reference.infrastructure import (
    UINT_MAX,
    crowds,
    hub_profile,
    hubs,
    world,
    world_identity,
)
from mule_pattern_learner.testing.world import DAY_MS, HOUR_MS, Mule, T, WorldBuilder
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
from mule_pattern_learner.tigergraph.queries.mark_crowds import Crowd, Crowds
from mule_pattern_learner.tigergraph.queries.read_hubs import HubAccount, Hubs, Leaf
from mule_pattern_learner.tigergraph.queries.read_world import World
from mule_pattern_learner.tigergraph.queries.rows import PAGE_LIMIT, QueryRefused

D, H = DAY_MS, HOUR_MS
LOOKBACK = 90 * D


def _scorable(graph: ReferenceGraph, t: int = T) -> dict[str, str]:
    rows = graph.run(LIST_ACCOUNTS, list_accounts.params(t, "", PAGE_LIMIT))
    return {row.account_id: row.owner_id for row in list_accounts.parse(rows, "", PAGE_LIMIT)}


def _hubs(graph: ReferenceGraph, t: int = T) -> Hubs:
    return read_hubs.parse(graph.run(READ_HUBS, read_hubs.params(t)))


def _crowds(graph: ReferenceGraph, t: int = T) -> Crowds:
    return mark_crowds.parse(graph.run(MARK_CROWDS, mark_crowds.params(t)))


def _crowds_of(hubs: Hubs) -> Crowds:
    """read_hubs' device and IP leaves, as mark_crowds prints the same crowds."""
    return Crowds(
        tuple(Crowd(**asdict(leaf)) for leaf in hubs.devices),
        tuple(Crowd(**asdict(leaf)) for leaf in hubs.ips),
    )


# The core.


def _small() -> WorldBuilder:
    """Two customers of one home paying each other, with another bank's account."""
    w = WorldBuilder()
    w.household(["ann", "ben"], T - 400 * D, "d_home", "i_home")
    w.external("x_out", T - 400 * D)
    w.pay("x_1", "a_ann", "a_ben", T - 2 * D, 10.0, "ach", device="d_home", ip="i_home")
    w.pay("z_1", "a_ben", "x_out", T - D, 5.0, "zelle", device="d_home")
    return w


def _planted(plant: Callable[[WorldBuilder], object]) -> WorldBuilder:
    w = _small()
    plant(w)
    return w


PLANTED: list[tuple[str, Callable[[WorldBuilder], object], str]] = [
    ("a repeated id", lambda w: w.party("p_ann", T - 9 * D), "Party repeats an id"),
    (
        "an edge to an unknown vertex",
        lambda w: w.rows["Party_Owns_Account"].append(("p_nobody", "a_ann", T - D, 0)),
        "an unknown vertex",
    ),
    (
        "a repeated link of one pair and start",
        lambda w: w.device("p_ann", "d_home", T - 400 * D, T - D),
        "repeats a link of one pair and start",
    ),
    (
        "a payment seen on two devices",
        lambda w: w.rows["Transaction_Used_Device"].append(("x_1", "d_home", T - 2 * D)),
        "Transaction_Used_Device repeats a payment",
    ),
    (
        "a payment without a receiver",
        lambda w: w.rows["Transaction_To_Account"].clear(),
        "x_1 needs one sender and one receiver",
    ),
    (
        "an event edge with another time",
        lambda w: w.rows["Transfer_Used_IP"].append(("z_1", "i_home", T)),
        "another time",
    ),
    (
        "a second ownership tenure",
        lambda w: w.link("Party_Owns_Account", "p_ann", "a_ann", T - 500 * D, T - 450 * D),
        "a_ann has 2 ownership rows",
    ),
    (
        "a second owner",
        lambda w: w.link("Party_Owns_Account", "p_ben", "a_ann", T - 400 * D),
        "a_ann has 2 ownership rows",
    ),
    (
        "a payment id shared by the two payment types",
        lambda w: w.pay("x_1", "a_ben", "a_ann", T - 3 * D, 1.0, "zelle"),
        "share the id x_1",
    ),
]


@pytest.mark.parametrize(
    ("plant", "message"), [(p, m) for _, p, m in PLANTED], ids=[n for n, _, _ in PLANTED]
)
def test_the_graph_refuses_tables_tigergraph_would_hold_otherwise(
    plant: Callable[[WorldBuilder], object], message: str
) -> None:
    assert _small().graph().payments.keys() == {"x_1", "z_1"}
    with pytest.raises(ValueError, match=message):
        _planted(plant).graph()


def test_the_graph_refuses_a_table_set_or_columns_unlike_the_contract() -> None:
    tables = _small().tables()
    with pytest.raises(ValueError, match="needs the tables"):
        ReferenceGraph({name: table for name, table in tables.items() if name != "Token"})
    reordered = {**tables, "Party": tables["Party"][["joined_ms", "id"]]}
    with pytest.raises(ValueError, match="Party has the columns"):
        ReferenceGraph(reordered)


def test_the_graph_checks_every_value_against_its_columns_type() -> None:
    tables = _small().tables()
    parties = tables["Party"]
    # A column of its contract dtype is checked whole: no negative UINT, no missing string.
    for planted, column in (
        (parties.assign(joined_ms=[-1, 5]), "Party.joined_ms holds a value that is not a UINT"),
        (
            parties.assign(id=pd.Series(["p_ann", None], dtype="str")),
            "Party.id holds a value that is not a STRING",
        ),
    ):
        with pytest.raises(ValueError, match=column):
            ReferenceGraph({**tables, "Party": planted})
    # Any other column value by value: digits as text are no UINT, while Python objects of the
    # right types read as the contract's dtypes do.
    with pytest.raises(ValueError, match="Party.joined_ms holds '"):
        ReferenceGraph({**tables, "Party": parties.astype({"joined_ms": "str"})})
    # A STRING column holding numbers is no STRING: the contract's dtype for it is pandas 3's
    # string dtype, which is not object, so such a column is never read whole (pyproject.toml
    # requires pandas 3 for this).
    with pytest.raises(ValueError, match="Party.id holds 1, not a STRING"):
        ReferenceGraph({**tables, "Party": parties.assign(id=pd.Series([1, 2], dtype=object))})
    as_objects = ReferenceGraph({**tables, "Party": parties.astype(object)})
    assert world(as_objects) == world(_small().graph())


@pytest.mark.parametrize("amount", [float("nan"), float("inf"), float("-inf")])
def test_the_graph_refuses_an_amount_that_is_not_finite(amount: float) -> None:
    tables = _small().tables()
    for name, rows in (("Payment_Transaction", 1), ("Zelle_Transfer", 1)):
        planted = tables[name].assign(amount=[amount] * rows)
        # The whole column of its contract dtype, and the same column as Python objects.
        with pytest.raises(ValueError, match=f"{name}.amount holds a value that is not a finite"):
            ReferenceGraph({**tables, name: planted})
        with pytest.raises(ValueError, match=f"{name}.amount holds .*, not a finite DOUBLE"):
            ReferenceGraph({**tables, name: planted.astype(object)})
    # A frame made from rows is refused as it is made.
    with pytest.raises(ValueError, match="Zelle_Transfer.amount holds .*, not a finite DOUBLE"):
        frame("Zelle_Transfer", [("z_1", amount, T)])


def test_the_graph_indexes_payments_owners_and_links() -> None:
    graph = _small().graph()
    x_1, z_1 = graph.payments["x_1"], graph.payments["z_1"]
    assert (x_1.kind, x_1.rail, x_1.cents, x_1.device, x_1.ip) == (
        "Payment_Transaction",
        "ach",
        1_000,
        "d_home",
        "i_home",
    )
    assert (z_1.kind, z_1.rail, z_1.sender, z_1.receiver, z_1.ip) == (
        "Zelle_Transfer",
        "zelle",
        "a_ben",
        "x_out",
        None,
    )
    assert graph.ownership["a_ann"] == Link("p_ann", "a_ann", T - 400 * D, 0)
    assert "x_out" not in graph.ownership and [link.target for link in graph.owned["p_ben"]] == [
        "a_ben"
    ]
    assert [link.target for link in graph.enrolled[IP]["p_ann"]] == ["i_home"]
    assert [link.source for link in graph.users[DEVICE]["d_home"]] == ["p_ann", "p_ben"]
    assert graph.joined_ms == {"p_ann": T - 400 * D, "p_ben": T - 400 * D}
    assert graph.ids["Device"] == ["d_home"] and graph.ids["Account"] == ["a_ann", "a_ben", "x_out"]


def test_an_amount_is_floor_of_amount_times_100_plus_a_half_in_doubles() -> None:
    # 1.005 x 100 is 100.49999999999999 in doubles, so 100 cents, not 101; 2.675 x 100 is 267.5.
    assert (cents(1.005), cents(2.675), cents(0.125), cents(10.0)) == (100, 268, 13, 1_000)


def test_every_bound_of_the_view_is_strictly_before_t() -> None:
    w = WorldBuilder()
    w.customer("old", T - 400 * D)
    w.customer("new", T)
    w.customer("gone", T - 400 * D, closed_ms=T - 1)
    w.customer("closing", T - 400 * D, closed_ms=T)
    w.customer("yet", T - 1)
    w.customer("peer", T - 400 * D)
    for payment_id, ts in (("x_start", T - LOOKBACK), ("x_early", T - LOOKBACK - 1), ("x_at", T)):
        w.pay(payment_id, "a_old", "a_peer", ts, 1.0, "ach", device="d_1")
    w.pay("x_self", "a_old", "a_old", T - 1, 1.0, "internal", device="d_1")
    view = w.graph().as_of(T)
    # An account exists when it opened before t, and is open unless it closed by then.
    assert [view.exists(a) for a in ("a_old", "a_new", "a_yet")] == [True, False, True]
    assert [view.open(a) for a in ("a_old", "a_gone", "a_closing")] == [True, False, True]
    # A link is visible from before t until 90 days before t, and current until 30 days.
    for start, end, visible, current in (
        (T, 0, False, False),
        (T - 1, 0, True, True),
        (T - 100 * D, T - LOOKBACK, False, False),
        (T - 100 * D, T - LOOKBACK + 1, True, False),
        (T - 100 * D, T - 30 * D, True, False),
        (T - 100 * D, T - 30 * D + 1, True, True),
        (T - 100 * D, T + D, True, True),
    ):
        link = Link("p_old", "d_1", start, end)
        assert (view.visible(link), view.current(link)) == (visible, current), (start, end)
    # A payment counts from t - 90 days to just before t. A self-payment is read by no query
    # but the crowd count: payments and seen keep it, sent and received leave it out.
    assert [p.id for p in view.payments("a_old")] == ["x_start", "x_self"]
    assert [p.id for p in view.sent("a_old")] == ["x_start"]
    assert [p.id for p in view.received("a_peer")] == ["x_start"]
    assert view.received("a_old") == [] and view.sent("a_nobody") == []
    assert [p.id for p in view.seen(DEVICE, "d_1")] == ["x_start", "x_self"]
    assert view.owner("a_gone") == "p_gone" and view.owner("a_new") is None


def test_run_answers_an_installed_query_with_exactly_its_parameters() -> None:
    graph = _small().graph()
    with pytest.raises(ValueError, match="No query fetch_subgraphs"):
        graph.run("fetch_subgraphs", {})
    with pytest.raises(ValueError, match="takes the parameters"):
        graph.run(READ_HUBS, {"t_ms": T, "lookback_days": 90})
    with pytest.raises(ValueError, match="takes the parameters"):
        graph.run(LIST_ACCOUNTS, {"t_ms": T, "after_id": ""})
    # The keywords of QueryExecutor.run are taken and ignored.
    rows = graph.run(READ_WORLD, {}, timeout_s=1.0, attempts=1, timeout_retries=0)
    assert read_world.parse(rows) == world(graph)


@pytest.mark.parametrize(
    ("name", "params"),
    [
        (LIST_ACCOUNTS, {"t_ms": T + 1, "after_id": "", "page": 10}),
        (LIST_ACCOUNTS, {"t_ms": T, "after_id": "", "page": 0}),
        (LIST_ACCOUNTS, {"t_ms": T, "after_id": "", "page": PAGE_LIMIT + 1}),
        (READ_HUBS, {"t_ms": T + 1}),
        (READ_HUBS, {"t_ms": 88 * D}),
        (MARK_CROWDS, {"t_ms": T + 1}),
        (MARK_CROWDS, {"t_ms": 90 * D}),
        (PROFILE_HUBS, {"t_ms": T - 999}),
        (PROFILE_HUBS, {"t_ms": 90 * D}),
        (READ_LABELS, {"after_id": "", "page": 0}),
        (READ_ANSWER_KEY, {"after_id": "", "page": PAGE_LIMIT + 1}),
    ],
)
def test_each_query_refuses_invalid_parameters_with_a_status_row(
    name: str, params: dict[str, object]
) -> None:
    assert _small().graph().run(name, params) == [{"status": "invalid_parameters"}]


def test_the_reference_graph_answers_any_whole_second_after_the_lookback() -> None:
    graph = _small().graph()
    # A Tuesday at 13:47:21 and a Sunday a second before midnight are times as a Monday at
    # midnight is, and the first second after the lookback is the earliest.
    for t in (T + D + 49_641_000, T + 7 * D - 1_000, LOOKBACK + 1_000):
        assert valid_time(t)
        assert read_hubs.parse(graph.run(READ_HUBS, {"t_ms": t})) == _hubs(graph, t)
        assert mark_crowds.parse(graph.run(MARK_CROWDS, {"t_ms": t})) == _crowds(graph, t)
        profile_hubs.parse(graph.run(PROFILE_HUBS, {"t_ms": t}))  # a refusal would raise
    # Half a second, a millisecond, and the lookback's length itself are not.
    assert not any(valid_time(t) for t in (T + 500, T + 1, LOOKBACK, LOOKBACK - 1_000, 0))


# list_accounts.


def test_list_accounts_lists_internal_deposit_accounts_open_and_owned_at_t() -> None:
    w = WorldBuilder()
    w.customer("open", T - 100 * D)
    w.customer("opened_at_t", T)
    w.customer("closed_at_t", T - 100 * D, closed_ms=T)
    w.customer("closed_before", T - 100 * D, closed_ms=T - 1)
    w.customer("card", T - 100 * D, account_type="credit")
    w.account("a_second", "p_open", T - 50 * D)
    w.account("a_unowned", None, T - 100 * D)
    w.external("x_deposit", T - 100 * D, account_type="deposit")
    w.customer("mule", T - 100 * D, mule=Mule(T - 20 * D, T - 10 * D, 4, identified=True))
    # Open deposit accounts of p_open whose one ownership link is written by hand, so that its
    # visibility is the only thing that differs: it starts at t, or ended exactly 90 days
    # before t, or started a millisecond before t, or ended a millisecond after the lookback's
    # start.
    for name, start, end in (
        ("link_at_t", T, 0),
        ("link_ended_at_lookback_start", T - 100 * D, T - LOOKBACK),
        ("link_just_before_t", T - 1, 0),
        ("link_ending_just_after_lookback_start", T - 100 * D, T - LOOKBACK + 1),
    ):
        w.account(f"a_{name}", None, T - 100 * D)
        w.link("Party_Owns_Account", "p_open", f"a_{name}", start, end)
    graph = w.graph()
    # Opened exactly at t: not listed. Closed exactly at t: still open at t, listed. A credit
    # card, an ownerless deposit account and another bank's are never scorable, and neither is
    # an account whose ownership link is not visible at t: one starting at t or ending at
    # t - 90 days is not, one starting at t - 1 or ending at t - 90 days + 1 is.
    assert _scorable(graph) == {
        "a_closed_at_t": "p_closed_at_t",
        "a_link_ending_just_after_lookback_start": "p_open",
        "a_link_just_before_t": "p_open",
        "a_mule": "p_mule",
        "a_open": "p_open",
        "a_second": "p_open",
    }
    # A week later the account opened at t is listed, and the one closed at t is not. The link
    # that started at t is visible, and the one that ended a millisecond after the lookback's
    # start is not, since its lookback now starts a week later.
    assert set(_scorable(graph, T + 7 * D)) == {
        "a_link_at_t",
        "a_link_just_before_t",
        "a_mule",
        "a_open",
        "a_opened_at_t",
        "a_second",
    }


def test_list_accounts_pages_by_strictly_increasing_id() -> None:
    w = WorldBuilder()
    for name in ("c", "a", "d", "b"):
        w.customer(name, T - 100 * D)
    graph = w.graph()
    pages: list[list[str]] = []
    after = ""
    while True:
        rows = graph.run(LIST_ACCOUNTS, list_accounts.params(T, after, 3))
        page = [row.account_id for row in list_accounts.parse(rows, after, 3)]
        pages.append(page)
        if not page:
            break
        after = page[-1]
    assert pages == [["a_a", "a_b", "a_c"], ["a_d"], []]


# read_labels and read_answer_key.


def test_read_labels_prints_the_identified_mules_with_their_activity() -> None:
    w = WorldBuilder()
    w.customer("caught", T - 100 * D, mule=Mule(T - 30 * D, T - 20 * D, 1, identified=True))
    w.customer("hidden", T - 100 * D, mule=Mule(T - 30 * D, T - 20 * D, 1))
    w.customer("plain", T - 100 * D)
    w.customer("early", T - 100 * D, mule=Mule(T - 60 * D, T - 60 * D, identified=True))
    graph = w.graph()
    rows = graph.run(READ_LABELS, read_labels.params("", PAGE_LIMIT))
    found = {
        (m.account_id, m.mule_onset_ms, m.mule_last_ms)
        for m in read_labels.parse(rows, "", PAGE_LIMIT)
    }
    # Only pu_label 1: a hidden mule's dates never leave the graph. Onset equal to last is valid.
    assert found == {("a_caught", T - 30 * D, T - 20 * D), ("a_early", T - 60 * D, T - 60 * D)}


@pytest.mark.parametrize(
    ("onset", "last"),
    [(T - 10 * D, T - 20 * D), (0, T - 20 * D)],
    ids=["onset after last", "no onset"],
)
def test_read_labels_refuses_a_page_with_an_activity_out_of_order(onset: int, last: int) -> None:
    w = WorldBuilder()
    w.customer("a", T - 100 * D, mule=Mule(T - 30 * D, T - 20 * D, identified=True))
    w.customer("b", T - 100 * D, mule=Mule(onset, last, identified=True))
    w.customer("c", T - 100 * D, mule=Mule(T - 30 * D, T - 20 * D, identified=True))
    graph = w.graph()
    # The first page of one mule holds a_a alone and is answered; a page holding a_b is not.
    first = read_labels.parse(graph.run(READ_LABELS, read_labels.params("", 1)), "", 1)
    assert [mule.account_id for mule in first] == ["a_a"]
    refused = graph.run(READ_LABELS, read_labels.params("a_a", 2))
    assert refused == [{"status": "invalid_activity", "accounts": ["a_b"]}]
    with pytest.raises(QueryRefused, match="invalid_activity"):
        read_labels.parse(refused, "a_a", 2)


def test_read_answer_key_prints_every_internal_accounts_truth() -> None:
    w = WorldBuilder()
    w.customer("hidden", T - 100 * D, mule=Mule(T - 30 * D, T - 20 * D, 6))
    w.customer("plain", T + 5 * D, account_type="credit", closed_ms=T + 9 * D)
    w.account("a_ledger", None, T - 300 * D, account_type="gl")
    w.external("x_other", T - 100 * D)
    graph = w.graph()
    rows = graph.run(READ_ANSWER_KEY, answer_key.params("", PAGE_LIMIT))
    truths = answer_key.parse(rows, "", PAGE_LIMIT)
    # Another bank's account is left out; owned says only whether any ownership link exists.
    assert [
        (t.account_id, t.is_mule, t.mule_ring_id, t.mule_onset_ms, t.mule_last_ms) for t in truths
    ] == [
        ("a_hidden", 1, 6, T - 30 * D, T - 20 * D),
        ("a_ledger", 0, -1, 0, 0),
        ("a_plain", 0, -1, 0, 0),
    ]
    assert [(t.opened_ms, t.account_type, t.owned) for t in truths] == [
        (T - 100 * D, "deposit", True),
        (T - 300 * D, "gl", False),
        (T + 5 * D, "credit", True),
    ]


# read_hubs: account hubs.


def _hub_world(plant: Callable[[WorldBuilder], object]) -> WorldBuilder:
    """A shop paid once by each of 200 customers 10 days before t, then the planted change."""
    w = WorldBuilder()
    w.customer("shop", T - 900 * D)
    for index in range(200):
        account = w.customer(f"c{index:03d}", T - 400 * D)
        w.pay(f"x_c{index:03d}", account, "a_shop", T - 10 * D, 5.0, "card")
    w.customer("new", T - 400 * D)
    plant(w)
    return w


HUB_CASES: list[tuple[str, Callable[[WorldBuilder], object], int | None]] = [
    ("200 customers", lambda w: None, None),
    ("a 201st customer", lambda w: w.pay("x_n", "a_new", "a_shop", T - 10 * D, 5.0, "card"), 201),
    ("paid by zelle", lambda w: w.pay("z_n", "a_new", "a_shop", T - 10 * D, 5.0, "zelle"), 201),
    ("a payment the shop sent", lambda w: w.pay("x_n", "a_shop", "a_new", T - D, 5.0, "ach"), 201),
    (
        "at the lookback's start",
        lambda w: w.pay("x_n", "a_new", "a_shop", T - LOOKBACK, 5.0, "ach"),
        201,
    ),
    (
        "before the lookback",
        lambda w: w.pay("x_n", "a_new", "a_shop", T - LOOKBACK - 1, 5.0, "ach"),
        None,
    ),
    ("at t", lambda w: w.pay("x_n", "a_new", "a_shop", T, 5.0, "ach"), None),
    (
        "a customer's second account",
        lambda w: (
            w.account("a_c000_2", "p_c000", T - 50 * D),
            w.pay("x_n", "a_c000_2", "a_shop", T - D, 5.0, "ach"),
        ),
        None,
    ),
    (
        "another bank's account",
        lambda w: (
            w.external("x_bank", T - 50 * D),
            w.pay("x_n", "x_bank", "a_shop", T - D, 5.0, "ach"),
        ),
        None,
    ),
    (
        "an ownerless ledger",
        lambda w: (
            w.account("g_fees", None, T - 50 * D, account_type="gl"),
            w.pay("x_n", "a_shop", "g_fees", T - D, 5.0, "internal"),
        ),
        None,
    ),
    ("a self-payment", lambda w: w.pay("x_n", "a_shop", "a_shop", T - D, 5.0, "internal"), None),
    (
        "a payer opened a millisecond before t, owned by a customer",
        lambda w: (
            w.account("a_late", None, T - 1),
            w.link("Party_Owns_Account", "p_new", "a_late", T - D),
            w.pay("x_n", "a_late", "a_shop", T - D, 5.0, "ach"),
        ),
        201,
    ),
    (
        "a payer opened at t, owned by a customer through a link from before t",
        lambda w: (
            w.account("a_future", None, T),
            w.link("Party_Owns_Account", "p_new", "a_future", T - D),
            w.pay("x_n", "a_future", "a_shop", T - D, 5.0, "ach"),
        ),
        None,
    ),
    (
        "a payee opened after t, owned by a customer through a link from before t",
        lambda w: (
            w.account("a_future", None, T + D),
            w.link("Party_Owns_Account", "p_new", "a_future", T - D),
            w.pay("x_n", "a_shop", "a_future", T - D, 5.0, "ach"),
        ),
        None,
    ),
    (
        "an account closed a millisecond after it paid at the lookback's start",
        lambda w: (
            w.customer("brief", T - 400 * D, closed_ms=T - LOOKBACK + 1),
            w.pay("x_n", "a_brief", "a_shop", T - LOOKBACK, 5.0, "ach"),
        ),
        201,
    ),
]


@pytest.mark.parametrize(
    ("plant", "customers"), [(p, c) for _, p, c in HUB_CASES], ids=[n for n, _, _ in HUB_CASES]
)
def test_an_account_hub_has_more_than_200_customers_on_its_payments_other_ends(
    plant: Callable[[WorldBuilder], object], customers: int | None
) -> None:
    graph = _hub_world(plant).graph()
    expected = () if customers is None else (HubAccount("a_shop", customers),)
    assert _hubs(graph) == Hubs(expected, (), ())


# read_hubs: crowds.


# A planted change of a crowd world: given the builder, enrol and seen.
CrowdPlant = Callable[[WorldBuilder, Callable[..., None], Callable[..., None]], object]


def _crowd_world(kind: str, plant: CrowdPlant) -> WorldBuilder:
    """A cafe's device or IP with 8 customers: 4 enrolled on it since 100 days before t, and 4
    whose deposit accounts paid a merchant on it 5 days before t; then the planted change.
    plant gets the builder, enrol(party, start, end=0) and seen(payment, sender, ts,
    receiver="x_shop")."""
    w = WorldBuilder()
    node = "d_cafe" if kind == DEVICE else "i_cafe"

    def enrol(party: str, start: int, end: int = 0) -> None:
        (w.device if kind == DEVICE else w.ip)(party, node, start, end)

    def seen(payment: str, sender: str, ts: int, receiver: str = "x_shop") -> None:
        if kind == DEVICE:
            w.pay(payment, sender, receiver, ts, 20.0, "card", device=node)
        else:
            w.pay(payment, sender, receiver, ts, 20.0, "card", ip=node)

    w.external("x_shop", T - 900 * D)
    for index in range(4):
        w.customer(f"e{index}", T - 400 * D)
        enrol(f"p_e{index}", T - 100 * D)
        w.customer(f"s{index}", T - 400 * D)
        seen(f"x_s{index}", f"a_s{index}", T - 5 * D)
    w.customer("n", T - 400 * D)
    plant(w, enrol, seen)
    return w


CROWD_CASES: list[tuple[str, CrowdPlant, bool]] = [
    ("8 customers", lambda w, enrol, seen: None, False),
    ("a ninth enrolled customer", lambda w, enrol, seen: enrol("p_n", T - 100 * D), True),
    ("a ninth sender", lambda w, enrol, seen: seen("x_n", "a_n", T - 5 * D), True),
    ("a link starting at t", lambda w, enrol, seen: enrol("p_n", T), False),
    ("a link starting just before t", lambda w, enrol, seen: enrol("p_n", T - 1), True),
    (
        "a link ending at t - 90 days",
        lambda w, enrol, seen: enrol("p_n", T - 100 * D, T - LOOKBACK),
        False,
    ),
    (
        "a link ending a millisecond later",
        lambda w, enrol, seen: enrol("p_n", T - 100 * D, T - LOOKBACK + 1),
        True,
    ),
    ("a customer enrolled twice", lambda w, enrol, seen: enrol("p_e0", T - 50 * D), False),
    ("an enrolled customer sending", lambda w, enrol, seen: seen("x_n", "a_e0", T - D), False),
    ("a sender sending again", lambda w, enrol, seen: seen("x_n", "a_s0", T - D), False),
    ("a payment seen at t", lambda w, enrol, seen: seen("x_n", "a_n", T), False),
    (
        "a payment seen at the lookback's start",
        lambda w, enrol, seen: seen("x_n", "a_n", T - LOOKBACK),
        True,
    ),
    (
        "a payment seen before the lookback",
        lambda w, enrol, seen: seen("x_n", "a_n", T - LOOKBACK - 1),
        False,
    ),
    (
        "the receiver of a payment seen on it",
        lambda w, enrol, seen: seen("x_n", "a_s0", T - D, "a_n"),
        False,
    ),
    (
        "a credit card user",
        lambda w, enrol, seen: (
            w.customer("card", T - 400 * D, account_type="credit"),
            seen("x_n", "a_card", T - D),
        ),
        False,
    ),
    (
        "an ownerless deposit account",
        lambda w, enrol, seen: (
            w.account("a_loose", None, T - 400 * D),
            seen("x_n", "a_loose", T - D),
        ),
        False,
    ),
    (
        "another bank's deposit account",
        lambda w, enrol, seen: (
            w.external("x_bank", T - 400 * D, account_type="deposit"),
            seen("x_n", "x_bank", T - D, "a_n"),
        ),
        False,
    ),
    (
        "a sender opened a millisecond before t, owned by a customer",
        lambda w, enrol, seen: (
            w.account("a_late", None, T - 1),
            w.link("Party_Owns_Account", "p_n", "a_late", T - D),
            seen("x_n", "a_late", T - D),
        ),
        True,
    ),
    (
        "a sender opened at t, owned by a customer through a link from before t",
        lambda w, enrol, seen: (
            w.account("a_future", None, T),
            w.link("Party_Owns_Account", "p_n", "a_future", T - D),
            seen("x_n", "a_future", T - D),
        ),
        False,
    ),
    (
        "a sender opened after t, owned by a customer through a link from before t",
        lambda w, enrol, seen: (
            w.account("a_future", None, T + D),
            w.link("Party_Owns_Account", "p_n", "a_future", T - D),
            seen("x_n", "a_future", T - D),
        ),
        False,
    ),
    (
        "a self-payment, which the crowd count reads",
        lambda w, enrol, seen: seen("x_n", "a_n", T - D, "a_n"),
        True,
    ),
]


@pytest.mark.parametrize("kind", [DEVICE, IP])
@pytest.mark.parametrize(
    ("plant", "crowd"), [(p, c) for _, p, c in CROWD_CASES], ids=[n for n, _, _ in CROWD_CASES]
)
def test_a_crowd_has_more_than_8_customers_by_the_one_account_rule(
    kind: str, plant: CrowdPlant, crowd: bool
) -> None:
    graph = _crowd_world(kind, plant).graph()
    leaves = (Leaf("d_cafe" if kind == DEVICE else "i_cafe", 9, False, False),) if crowd else ()
    expected = Hubs((), leaves, ()) if kind == DEVICE else Hubs((), (), leaves)
    assert _hubs(graph) == expected
    assert _crowds(graph) == _crowds_of(expected)


def _busy_node_world(kind: str, customers: int, enrolled: int) -> WorldBuilder:
    """A device or IP of customers customers: the first enrolled of them enrolled on it since
    100 days before t, and each of the others the owner of a deposit account that paid a
    merchant of its own on it 5 days before t, so no account is paid by more than one."""
    w = WorldBuilder()
    node = "d_busy" if kind == DEVICE else "i_busy"
    for index in range(customers):
        name = f"c{index:03d}"
        account = w.customer(name, T - 400 * D)
        if index < enrolled:
            (w.device if kind == DEVICE else w.ip)(f"p_{name}", node, T - 100 * D)
        else:
            w.external(f"x_shop_{name}", T - 900 * D)
            w.pay(
                f"x_{name}",
                account,
                f"x_shop_{name}",
                T - 5 * D,
                20.0,
                "card",
                device=node if kind == DEVICE else None,
                ip=node if kind == IP else None,
            )
    return w


@pytest.mark.parametrize("kind", [DEVICE, IP])
@pytest.mark.parametrize("enrolled", [0, 100], ids=["observed only", "half enrolled"])
@pytest.mark.parametrize(("customers", "hub"), [(200, False), (201, True)])
def test_a_device_or_ip_is_a_hub_with_more_than_200_customers(
    kind: str, enrolled: int, customers: int, hub: bool
) -> None:
    graph = _busy_node_world(kind, customers, enrolled).graph()
    leaf = Leaf("d_busy" if kind == DEVICE else "i_busy", customers, enrolled == 0, hub)
    # Either way it is a crowd, with more than 8 customers; it is a hub only with more than 200.
    expected = Hubs((), (leaf,), ()) if kind == DEVICE else Hubs((), (), (leaf,))
    assert _hubs(graph) == expected
    assert _crowds(graph) == _crowds_of(expected)


def test_a_leaf_says_whether_it_is_observed_only_and_a_hub() -> None:
    w = WorldBuilder()
    w.external("x_atm", T - 900 * D, account_type="gl")
    # Nine customers withdraw on a terminal nobody registered: a crowd, observed only.
    for index in range(9):
        account = w.customer(f"s{index}", T - 400 * D)
        w.pay(f"x_s{index}", account, "x_atm", T - 3 * D, 40.0, "cash", device="d_atm")
    # 201 customers on a carrier's IP, one of them registered on it: a crowd and a hub.
    w.customer("r", T - 400 * D)
    w.ip("p_r", "i_carrier", T - 60 * D)
    for index in range(200):
        account = w.customer(f"m{index:03d}", T - 400 * D)
        w.pay(f"x_m{index:03d}", account, "x_atm", T - 2 * D, 40.0, "card", ip="i_carrier")
    graph = w.graph()
    # The terminal's account is an account hub too, paid by the owners of all 209 accounts.
    assert _hubs(graph) == Hubs(
        (HubAccount("x_atm", 209),),
        (Leaf("d_atm", 9, True, False),),
        (Leaf("i_carrier", 201, False, True),),
    )
    # The printed rows are those hubs() computes, and mark_crowds prints the same crowds.
    assert hubs(graph, T) == _hubs(graph)
    assert crowds(graph, T) == _crowds(graph) == _crowds_of(_hubs(graph))


# profile_hubs.


def test_profile_hubs_counts_every_existing_account_and_every_device_and_ip_with_a_customer() -> (
    None
):
    w = WorldBuilder()
    w.household(["ann", "ben"], T - 400 * D, "d_home", "i_home")
    w.customer("cat", T - 400 * D)
    w.customer("dan", T)
    w.customer("idle", T - 400 * D)
    w.external("x_bank", T - 400 * D)
    for index, sender in enumerate(("a_ben", "a_cat", "x_bank")):
        w.pay(f"x_{index}", sender, "a_ann", T - 4 * D, 1.0, "ach")
    w.pay("x_atm", "a_cat", "x_bank", T - 2 * D, 1.0, "cash", device="d_atm")
    # A device whose only link ended before the lookback has no customer, and is left out.
    w.device("p_idle", "d_old", T - 300 * D, T - 200 * D)
    graph = w.graph()
    profile = profile_hubs.parse(graph.run(PROFILE_HUBS, profile_hubs.params(T)))
    # The accounts that exist at t, a_dan opened at t left out: a_ann has p_ben and p_cat
    # (x_bank is ownerless); a_ben has p_ann; a_cat has p_ann (x_bank again); x_bank has
    # p_ann and p_cat; a_idle has none.
    assert profile.accounts == {0: 1, 1: 2, 2: 2}
    assert hub_profile(graph, T) == profile
    # d_home and i_home: p_ann and p_ben, registered; d_atm: p_cat, seen only.
    assert (profile.registered_devices, profile.observed_devices) == ({2: 1}, {1: 1})
    assert (profile.registered_ips, profile.observed_ips) == ({2: 1}, {})
    assert graph.run(PROFILE_HUBS, profile_hubs.params(T))[0]["accounts"] == {
        "0": 1,
        "1": 2,
        "2": 2,
    }


# read_world.


def test_read_world_counts_and_sums_every_table_by_hand() -> None:
    w = WorldBuilder()
    w.party("p_a", 1_000_500)
    w.party("p_b", 2_000_999)
    w.account("a_1", "p_a", 3_000_000)
    w.account("a_2", "p_b", 4_000_400, closed_ms=9_000_900)
    w.external("x_1", 5_000_000)
    w.device("p_a", "d_1", 3_500_700)
    w.device("p_b", "d_1", 4_100_000, 8_000_000)
    w.ip("p_a", "i_1", 3_600_000)
    w.token("t_1", "p_a", "a_1", 3_700_000)
    w.pay("x_p1", "a_1", "a_2", 6_000_123, 1.005, "ach", device="d_1", ip="i_1")
    w.pay("x_p2", "a_2", "x_1", 7_000_456, 0.125, "card")
    w.pay("z_p1", "a_1", "a_2", 8_000_789, 2.675, "zelle")
    found = read_world.parse(w.graph().run(READ_WORLD, read_world.params()))
    ones = dict.fromkeys(
        (
            "Party_Uses_Token",
            "Token_Bound_To_Account",
            "Party_Uses_IP",
            "Transaction_Used_Device",
            "Transaction_Used_IP",
            "Transfer_From_Account",
            "Transfer_To_Account",
        ),
        1,
    )
    zeros = dict.fromkeys(
        ("Transfer_From_Token", "Transfer_To_Token", "Transfer_Used_Device", "Transfer_Used_IP"), 0
    )
    pairs = {"Party_Owns_Account": 2, "Transaction_From_Account": 2, "Transaction_To_Account": 2}
    assert found == World(
        vertices={
            "Party": 2,
            "Account": 3,
            "Token": 1,
            "Device": 1,
            "IP": 1,
            "Payment_Transaction": 2,
            "Zelle_Transfer": 1,
        },
        edges={**ones, **zeros, **pairs, "Party_Uses_Device": 2},
        account_types={"deposit": 2, "credit": 0, "brokerage": 0, "gl": 0, "unknown": 1},
        payment_rails={"ach": 1, "card": 1, "cash": 0, "check": 0, "internal": 0, "unknown": 0},
        external_accounts=1,
        # Each time rounded down to whole seconds, then summed.
        seconds={
            "Party.joined_ms": 1_000 + 2_000,
            "Account.opened_ms": 3_000 + 4_000 + 5_000,
            "Account.closed_ms": 9_000,
            "Payment_Transaction.event_ts_ms": 6_000 + 7_000,
            "Zelle_Transfer.event_ts_ms": 8_000,
            "Party_Owns_Account.valid_from_ms": 3_000 + 4_000,
            "Party_Owns_Account.valid_to_ms": 9_000,
            "Party_Uses_Token.valid_from_ms": 3_700,
            "Party_Uses_Token.valid_to_ms": 0,
            "Token_Bound_To_Account.valid_from_ms": 3_700,
            "Token_Bound_To_Account.valid_to_ms": 0,
            "Party_Uses_Device.valid_from_ms": 3_500 + 4_100,
            "Party_Uses_Device.valid_to_ms": 8_000,
            "Party_Uses_IP.valid_from_ms": 3_600,
            "Party_Uses_IP.valid_to_ms": 0,
        },
        # 1.005 is 100 cents in doubles, 0.125 is 13 and 2.675 is 268.
        cents={"Payment_Transaction": 100 + 13, "Zelle_Transfer": 268},
        # Every source has one edge of each type, but d_1 is the target of two enrolments.
        squared_degrees={**ones, **zeros, **pairs, "Party_Uses_Device": 2},
        squared_target_degrees={**ones, **zeros, **pairs, "Party_Uses_Device": 4},
        degree_products={**ones, **zeros, **pairs, "Party_Uses_Device": 1 * 2 + 1 * 2},
        first_joined_ms=1_000_500,
        first_event_ms=6_000_123,
        last_event_ms=8_000_789,
    )


def _degrees(moved: bool) -> WorldBuilder:
    """Enrolments p_1 on d_1 and d_2, p_2 on d_3 and p_4 on d_3; or, moved, p_1's link to d_1
    moved to p_2, so p_1 has one and p_2 two."""
    w = WorldBuilder()
    for party in ("p_1", "p_2", "p_4"):
        w.party(party, 1_000)
    for party, device in (
        ("p_2" if moved else "p_1", "d_1"),
        ("p_1", "d_2"),
        ("p_2", "d_3"),
        ("p_4", "d_3"),
    ):
        w.device(party, device, 5_000)
    return w


def test_read_world_changes_when_a_link_moves_between_vertices_of_degrees_2_and_1() -> None:
    before, after = world(_degrees(False).graph()), world(_degrees(True).graph())
    kind = "Party_Uses_Device"
    # Out-degrees (2, 1, 1) become (1, 2, 1) and in-degrees stay (1, 1, 2): squared sums 6.
    assert before.squared_degrees[kind] == after.squared_degrees[kind] == 4 + 1 + 1
    assert before.squared_target_degrees[kind] == after.squared_target_degrees[kind] == 1 + 1 + 4
    # Over the edges, out times in: 2x1 + 2x1 + 1x2 + 1x2 before, 1x1 + 2x1 + 2x2 + 1x2 after.
    assert (before.degree_products[kind], after.degree_products[kind]) == (8, 9)
    assert replace(after, degree_products=before.degree_products) == before
    assert world_identity(_degrees(False).graph()) != world_identity(_degrees(True).graph())


def test_read_world_of_an_empty_world_holds_tigergraphs_starting_values() -> None:
    found = world(WorldBuilder().graph())
    assert (found.first_joined_ms, found.first_event_ms, found.last_event_ms) == (
        UINT_MAX,
        UINT_MAX,
        0,
    )
    assert set(found.vertices.values()) == {0} and found.external_accounts == 0
