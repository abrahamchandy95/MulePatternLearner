"""The reference answer of read_counterparties, B2, on worlds worked out by hand from
docs/features.md ("Account node", columns 22 to 33): the look-alike fixtures, the windows'
bounds, the whole-second cut of the new payers' first payments, the hub parameter, the
refusals, and the leak tests over B2.

Every answer goes through ReferenceGraph.run and the query's own parser, so it has the shape
TigerGraph prints and the parser's checks hold of it. Amounts are in cents; a share is cents
over cents.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
import json

import pytest

from mule_pattern_learner.testing import fixtures
from mule_pattern_learner.testing.reference import ReferenceGraph
from mule_pattern_learner.testing.world import DAY_MS, HOUR_MS, T, WorldBuilder
from mule_pattern_learner.tigergraph.queries import READ_COUNTERPARTIES, read_counterparties
from mule_pattern_learner.tigergraph.queries.read_counterparties import Counterparties
from mule_pattern_learner.tigergraph.queries.rows import LOOKBACK_MS, QueryRefused

D, H, M = DAY_MS, HOUR_MS, fixtures.MINUTE_MS


def _rows(
    graph: ReferenceGraph,
    accounts: Sequence[str],
    hubs: Sequence[str] = fixtures.HUBS,
    t: int = T,
) -> tuple[Counterparties, ...]:
    found = graph.run(READ_COUNTERPARTIES, read_counterparties.params(t, accounts, hubs))
    return read_counterparties.parse(found, accounts)


def _existing(graph: ReferenceGraph, t: int = T) -> list[str]:
    """Every account that exists at t, which B2 may be asked for."""
    return [a for a in graph.ids["Account"] if graph.accounts[a].opened_ms < t]


# Each look-alike's B2 by hand: payers_30d, payees_30d, new internal payers over 7 and 30
# days, new external payers over 7 and 30 days, new payees over 7 and 30 days, the share of
# the 30 days' cents received from new payers, the most new payers' first payments in 24
# hours, and the shares of the 30 days' cents sent to the top payee and to hubs.
LOOK_ALIKES: dict[str, Counterparties] = {
    # Paid only by its employer, every two weeks for 90 days; pays its savings (120,000 over
    # 30 days), the grocer (300,000) and the utility (240,000, three bills), none of them new.
    # The grocer and the utility are hubs.
    "a_split": Counterparties(
        "a_split", 1, 3, 0, 0, 0, 0, 0, 0, 0.0, 0, 300_000 / 660_000, 540_000 / 660_000
    ),
    # Its employer (first 58 days before t) and its own savings, whose 10,000 at t - 8 d
    # 21:00, before t - 7 days, is its first payment in the lookback: new over 30 days only,
    # and internal. Pays savings 50,000, rent 140,000, and 12,000 at a shop first paid at
    # t - 8 d 22:00, also before t - 7 days; the shop is a hub.
    "a_saver": Counterparties(
        "a_saver", 2, 3, 0, 1, 0, 0, 0, 1, 10_000 / 330_000, 1, 140_000 / 202_000, 12_000 / 202_000
    ),
    # The treasury's refund at t - 12 d is its only payment in: a new external payer over 30
    # days, all 150,000; all 67,000 out went to one new payee, a hub mall.
    "a_sleeper": Counterparties("a_sleeper", 1, 1, 0, 0, 0, 1, 0, 1, 1.0, 1, 1.0, 1.0),
    # Three tenants who have paid since t - 64 days: none new. Pays the loan ledger 250,000,
    # a hub, and a plumber 30,000, new over 7 days.
    "a_landlord": Counterparties(
        "a_landlord", 3, 2, 0, 0, 0, 0, 1, 1, 0.0, 0, 250_000 / 280_000, 250_000 / 280_000
    ),
    # Paid only by its parent since t - 50 days. Pays the school 190,000, first paid at
    # t - 19 d (new over 30 days), and a hub cafe 10,500, first paid at t - 45 d.
    "a_student": Counterparties(
        "a_student", 1, 2, 0, 0, 0, 0, 0, 1, 0.0, 0, 190_000 / 200_500, 10_500 / 200_500
    ),
    # The new employer is its only payer in 30 days, external and new over 7 days. Pays rent
    # 110,000, the power biller 60,000 and a shop 8,000, the shop new over 7 days; the biller
    # and the shop are hubs.
    "a_newjob": Counterparties(
        "a_newjob", 1, 3, 0, 0, 1, 1, 1, 1, 1.0, 1, 110_000 / 178_000, 68_000 / 178_000
    ),
    # Its payroll and three victims: a_vic1 (internal) and x_vic2 (external) first pay in the
    # same second of t - 6 d 14:00, x_vic3 (external) 54 hours later; 350,000 of 530,000 came
    # from them. Pays the ATM 20,000 (new over 30 days), the shop 40,000 (not new) and the
    # collector 325,500 (new over 7 days); the ATM and the shop are hubs.
    "a_mule": Counterparties(
        "a_mule", 4, 3, 1, 1, 2, 2, 1, 2, 350_000 / 530_000, 2, 325_500 / 385_500, 60_000 / 385_500
    ),
    # Three internal mules, each first paying in the last 7 days: a_mule at t - 6 d 14:09 and
    # a_mule2 19 h 51 min later, then a_mule3 30 hours after that. Sends 450,000 to another
    # bank and 10,000 to a hub ATM, both new.
    "a_collector": Counterparties(
        "a_collector", 3, 2, 3, 3, 0, 0, 2, 2, 1.0, 2, 450_000 / 460_000, 10_000 / 460_000
    ),
    "a_newjob_old": Counterparties("a_newjob_old", 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0, 0.0, 0.0),
}


@pytest.fixture(scope="module")
def catalogue() -> ReferenceGraph:
    return fixtures.catalogue().graph()


@pytest.mark.parametrize("account", LOOK_ALIKES)
def test_each_look_alike_reads_as_worked_out_by_hand(
    catalogue: ReferenceGraph, account: str
) -> None:
    assert _rows(catalogue, [account]) == (LOOK_ALIKES[account],)


def test_a_request_answers_each_account_as_it_answers_it_alone(
    catalogue: ReferenceGraph,
) -> None:
    accounts = list(LOOK_ALIKES)
    assert _rows(catalogue, accounts) == tuple(LOOK_ALIKES[a] for a in sorted(accounts))


def test_the_hubs_are_those_of_the_parameter_alone(catalogue: ReferenceGraph) -> None:
    # With no hub, no cents went to hubs; with one the graph lacks, TigerGraph drops it.
    expected = LOOK_ALIKES["a_mule"]
    no_hubs = replace(expected, hub_payee_share_30d=0.0)
    assert _rows(catalogue, ["a_mule"], []) == (no_hubs,)
    assert _rows(catalogue, ["a_mule"], ["x_nowhere"]) == (no_hubs,)
    atm_only = replace(expected, hub_payee_share_30d=20_000 / 385_500)
    assert _rows(catalogue, ["a_mule"], ["x_mule_atm", "x_nowhere"]) == (atm_only,)


def test_a_payer_is_new_over_w_from_t_minus_w_to_just_before_t() -> None:
    w = WorldBuilder()
    w.customer("edge", fixtures.JOINED)
    for name in ("p30b", "p7", "pt"):
        w.customer(name, fixtures.JOINED)
    for account in ("x_p30", "x_p7b", "x_p24", "x_old", "x_q30", "x_q7b", "x_q31", "x_q90"):
        w.external(account, fixtures.OPENED)
    for payment_id, payer, ts, amount in (
        ("x_e_p30", "x_p30", T - 30 * D, 100.0),
        ("x_e_p30b_early", "a_p30b", T - 30 * D - 1, 50.0),
        ("x_e_p30b", "a_p30b", T - 10 * D, 20.0),
        ("x_e_p7b", "x_p7b", T - 7 * D - 1, 4.0),
        ("x_e_p7", "a_p7", T - 7 * D, 10.0),
        ("x_e_p24", "x_p24", T - 6 * D - 1, 5.0),
        ("x_e_old_before", "x_old", T - 100 * D, 40.0),
        ("x_e_old", "x_old", T - 20 * D, 30.0),
        ("x_e_pt", "a_pt", T, 90.0),
    ):
        w.pay(payment_id, payer, "a_edge", ts, amount, "ach")
    for payment_id, payee, ts, amount in (
        ("x_e_q30", "x_q30", T - 30 * D, 40.0),
        ("x_e_q31", "x_q31", T - 31 * D, 25.0),
        ("x_e_q7b", "x_q7b", T - 7 * D - 1, 10.0),
        ("x_e_q90_first", "x_q90", T - LOOKBACK_MS, 60.0),
        ("x_e_q90", "x_q90", T - 3 * D, 15.0),
    ):
        w.pay(payment_id, "a_edge", payee, ts, amount, "ach")
    # Received in the 30 days: 10,000 + 2,000 + 400 + 1,000 + 500 + 3,000 from six payers; a_pt
    # pays at t and is not read. New over 30 days: x_p30 first at exactly t - 30 d, x_p7b at a
    # millisecond before t - 7 d (new over 30 days only, external), a_p7 at exactly t - 7 d
    # (new over 7 days too, internal), x_p24 at t - 6 d - 1 ms (new over 7 days, external),
    # and x_old, whose earlier payment is before the lookback; not a_p30b, first a millisecond
    # before t - 30 d. They gave 14,900. Of their first payments, x_p7b's and a_p7's are a
    # millisecond apart, so a window of 24 hours holds two; x_p24's is exactly 24 hours after
    # x_p7b's, so none holds three. x_q30, first paid at exactly t - 30 d, and x_q7b, at a
    # millisecond before t - 7 d, are new payees (x_q7b over 30 days only); x_q90, first paid
    # at exactly t - 90 d, is not. x_q31, paid only 31 days before t, is no payee of the 30
    # days: it counts nowhere. Of 6,500 sent in 30 days, 4,000 went to x_q30, 1,000 to x_q7b
    # and 1,500 to the hub x_q90.
    expected = Counterparties(
        "a_edge", 6, 3, 1, 1, 1, 4, 0, 2, 14_900 / 16_900, 2, 4_000 / 6_500, 1_500 / 6_500
    )
    assert _rows(w.graph(), ["a_edge"], ["x_q90"]) == (expected,)


def test_the_new_payers_first_payments_are_cut_at_a_whole_second() -> None:
    w = WorldBuilder()
    w.customer("fan", fixtures.JOINED)
    for i in range(300):
        payer = w.external(f"x_f{i:03d}", fixtures.OPENED)
        minute = 43 if i == 44 else i
        w.pay(f"x_fan{i:03d}", payer, "a_fan", T - 10 * D + minute * M, 1.0, "ach")
    # 300 new external payers, one a minute from t - 10 days, x_f044 in the minute of x_f043.
    # The newest 256 first payments end in that tie: the newest left out is at minute 43, so
    # both payers of minute 43 are dropped, and the 255 kept, minutes 45 to 299, all fall in
    # 24 hours.
    expected = Counterparties("a_fan", 300, 0, 0, 0, 0, 300, 0, 0, 1.0, 255, 0.0, 0.0)
    assert _rows(w.graph(), ["a_fan"]) == (expected,)


@pytest.mark.parametrize(
    ("params", "status"),
    [
        ({"t_ms": T + 1}, "invalid_parameters"),
        ({"account_ids": []}, "invalid_parameters"),
        ({"account_ids": ["a_mule", "a_mule"]}, "duplicate_account"),
        ({"account_ids": ["a_nobody"]}, "unknown_account"),
        ({"account_ids": ["a_late"]}, "invisible_account"),
        ({"hub_accounts": [{"id": "x_mule_atm"}, {"id": "a_late"}]}, "invisible_hub"),
        ({"t_ms": T + 1, "account_ids": ["a_mule", "a_mule"]}, "invalid_parameters"),
        ({"account_ids": ["a_nobody", "a_nobody"]}, "duplicate_account"),
        ({"account_ids": ["a_late", "a_nobody"]}, "unknown_account"),
        (
            {"account_ids": ["a_late"], "hub_accounts": [{"id": "a_late"}]},
            "invisible_account",
        ),
    ],
    ids=[
        "no whole second",
        "no account",
        "an account twice",
        "an unknown account",
        "opened at t",
        "a hub opened at t",
        "no whole second before an account twice",
        "an account twice before an unknown one",
        "an unknown account before one opened at t",
        "an account opened at t before a hub opened at t",
    ],
)
def test_the_query_refuses_a_request_it_cannot_answer(
    params: dict[str, object], status: str
) -> None:
    w = fixtures.catalogue()
    w.customer("late", T)
    given = {"t_ms": T, "account_ids": ["a_mule"], "hub_accounts": [], **params}
    rows = w.graph().run(READ_COUNTERPARTIES, given)
    assert rows == [{"status": status}]
    with pytest.raises(QueryRefused, match=status):
        read_counterparties.parse(rows, ["a_mule"])


# The leak tests over B2.


def _printed(graph: ReferenceGraph) -> str:
    """The answer for every account that exists at t, as the bytes TigerGraph would send."""
    params = read_counterparties.params(T, _existing(graph), fixtures.HUBS)
    return json.dumps(graph.run(READ_COUNTERPARTIES, params))


def test_erasing_every_label_column_changes_no_answer() -> None:
    tables = fixtures.catalogue().tables()
    erased = fixtures.erase_labels(tables)
    assert not erased["Account"].equals(tables["Account"])
    assert _printed(ReferenceGraph(erased)) == _printed(ReferenceGraph(tables))


def test_moving_dates_of_record_before_the_lookback_changes_no_answer() -> None:
    tables = fixtures.catalogue().tables()
    moved = fixtures.move_dates_of_record(tables, T, LOOKBACK_MS)
    assert not moved["Account"].equals(tables["Account"])
    assert _printed(ReferenceGraph(moved)) == _printed(ReferenceGraph(tables))


def test_swapping_the_ids_of_same_second_payments_changes_no_value() -> None:
    w = fixtures.catalogue()
    # Two more payers of a_mule in one second, and two payments of one second to one payee.
    w.external("x_tie1", fixtures.OPENED)
    w.customer("tie2", fixtures.JOINED)
    w.pay("x_tie_b", "x_tie1", "a_mule", T - 2 * D + 5 * H, 70.0, "ach")
    w.pay("x_tie_a", "a_tie2", "a_mule", T - 2 * D + 5 * H, 30.0, "ach")
    w.pay("x_tie_d", "a_collector", "x_offshore", T - D + 9 * H, 5.0, "ach")
    w.pay("x_tie_c", "a_collector", "x_collector_atm", T - D + 9 * H, 5.0, "cash")
    tables = w.tables()
    swapped, original = fixtures.swap_same_second_ids(tables)
    assert any(new != old for new, old in original.items())
    assert _printed(ReferenceGraph(swapped)) == _printed(ReferenceGraph(tables))


def test_a_self_payment_is_read_nowhere() -> None:
    w = fixtures.catalogue()
    before = _printed(w.graph())
    w.pay("x_mule_self", "a_mule", "a_mule", T - 6 * D + 14 * H + M, 900.0, "internal")
    assert _printed(w.graph()) == before
