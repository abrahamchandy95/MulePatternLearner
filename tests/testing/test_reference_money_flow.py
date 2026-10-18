"""The reference answer of read_money_flow, B1, on worlds worked out by hand from
docs/features.md (B1's walk and the "Account node" table, columns 4 to 21, and the payment
edge's dwell_tracked, left_24h and dwell): the look-alike fixtures, the tie, boundary and cut
rules, the walk against full-history last-in, first-out matching, the refusals, and the leak
tests over B1.

Every answer goes through ReferenceGraph.run and the query's own parser, so it has the shape
TigerGraph prints and the parser's checks hold of it. Amounts are in cents and dwells in
seconds; a share is worked out as cents over cents.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
import json
import random

import pytest

from mule_pattern_learner.testing import fixtures
from mule_pattern_learner.testing.reference import ReferenceGraph
from mule_pattern_learner.testing.reference.graph import RECENT, Payment, second
from mule_pattern_learner.testing.reference.money_flow import walk
from mule_pattern_learner.testing.world import DAY_MS, HOUR_MS, T, WorldBuilder
from mule_pattern_learner.tigergraph.queries import READ_MONEY_FLOW, read_money_flow
from mule_pattern_learner.tigergraph.queries.read_money_flow import (
    MoneyFlow,
    MoneyFlows,
    TrackedInflow,
)
from mule_pattern_learner.tigergraph.queries.rows import LOOKBACK_MS, QueryRefused

D, H, M = DAY_MS, HOUR_MS, fixtures.MINUTE_MS
RAILS = ("ach", "card", "cash", "check", "internal", "unknown", "zelle")


def _flows(graph: ReferenceGraph, accounts: Sequence[str], t: int = T) -> MoneyFlows:
    rows = graph.run(READ_MONEY_FLOW, read_money_flow.params(t, accounts))
    return read_money_flow.parse(rows, accounts)


def _existing(graph: ReferenceGraph, t: int = T) -> list[str]:
    """Every account that exists at t, which B1 may be asked for."""
    return [a for a in graph.ids["Account"] if graph.accounts[a].opened_ms < t]


@dataclass(frozen=True)
class Hand:
    """An account's B1 values worked out by hand, in cents and seconds.

    - received, sent: the cents over the last 7, 30 and 90 days.
    - total: R, the cents of the lots of the span; within: the matched cents with a dwell
      under 15 minutes, 1 hour, 6 hours, 1 day, 3 days and 7 days; matched: every matched
      cent.
    - median_s: the dwell at which the pieces by ascending dwell first reach half of R.
    - rails: of the cents matched within a day, those that left by transfer, card and cash.
    - lots: each tracked payment's lot as (its cents matched within 24 hours, its cents, the
      dwell at which its pieces first reach half of it).
    """

    received: tuple[int, int, int]
    sent: tuple[int, int, int]
    total: int
    within: tuple[int, int, int, int, int, int]
    matched: int
    median_s: int | None
    rails: tuple[int, int, int]
    lots: dict[str, tuple[int, int, int | None]]

    def rows(self, account: str) -> tuple[MoneyFlow, list[TrackedInflow]]:
        """The row and tracked inflows these values make: each share is cents over cents."""
        r, day = self.total, self.within[3]
        left = [cents / r if r else 0.0 for cents in self.within]
        rails = [cents / day if day else 0.0 for cents in self.rails]
        row = MoneyFlow(
            account,
            *self.received,
            *self.sent,
            flow_present=r > 0,
            left_15m=left[0],
            left_1h=left[1],
            left_6h=left[2],
            left_1d=left[3],
            left_3d=left[4],
            left_7d=left[5],
            held_share=(r - self.matched) / r if r else 0.0,
            median_dwell_present=self.median_s is not None,
            median_dwell_ms=(self.median_s or 0) * 1_000,
            left_by_transfer=rails[0],
            left_by_card=rails[1],
            left_by_cash=rails[2],
            scan_steps=0,
        )
        tracked = [
            TrackedInflow(account, payment, left_24h / cents, half is not None, (half or 0) * 1_000)
            for payment, (left_24h, cents, half) in sorted(self.lots.items())
        ]
        return row, tracked


# Each look-alike's B1 by hand. The walk reads the payments of [t - 30 days, t).
LOOK_ALIKES: dict[str, Hand] = {
    # Salaries of 300,000 at t - 19 d 09:00 (S1) and t - 5 d 09:00 (S0); t - 33 d's bill of
    # 80,000 at t - 30 d 09:00 is in the span but finds no lot. From each salary: 60,000 to
    # savings after 20 min (1,200 s), 150,000 by card after 27 h (97,200 s), 80,000 by ACH
    # after exactly 3 d (259,200 s, not under 3 days); each keeps 10,000. The median: 120,000
    # by 1,200 s, then 420,000 by 97,200 s, past half of 600,000.
    "a_split": Hand(
        received=(300_000, 600_000, 7 * 300_000),
        sent=(290_000, 80_000 + 2 * 290_000, 7 * 290_000),
        total=600_000,
        within=(0, 120_000, 120_000, 120_000, 120_000 + 300_000, 580_000),
        matched=580_000,
        median_s=97_200,
        rails=(120_000, 0, 0),
        lots={"x_split_s0": (60_000, 300_000, 97_200), "x_split_s1": (60_000, 300_000, 97_200)},
    ),
    # Its savings only receive: 60,000 at t - 5 d and t - 19 d (09:20), five more before.
    "a_split_save": Hand(
        received=(60_000, 120_000, 7 * 60_000),
        sent=(0, 0, 0),
        total=120_000,
        within=(0, 0, 0, 0, 0, 0),
        matched=0,
        median_s=None,
        rails=(0, 0, 0),
        lots={"x_split_m0": (0, 60_000, None), "x_split_m1": (0, 60_000, None)},
    ),
    # Salary S of 320,000 at t - 28 d 09:00: 25,000 to savings after 35 h (126,000 s),
    # 140,000 rent after 49 h (176,400 s), 25,000 to savings after 371 h (1,335,600 s). The
    # 10,000 back from savings at t - 8 d 21:00 is the newest lot, so the 12,000 card payment
    # an hour later takes it all at exactly 3,600 s (not under an hour) and 2,000 of S after
    # 493 h. The median: 10,000, 35,000, then 175,000 at 176,400 s, past half of 330,000.
    "a_saver": Hand(
        received=(0, 330_000, 330_000 + 320_000),
        sent=(0, 202_000, 202_000 + 25_000 + 140_000),
        total=330_000,
        within=(0, 0, 10_000, 10_000, 10_000 + 25_000 + 140_000, 175_000),
        matched=25_000 + 140_000 + 25_000 + 10_000 + 2_000,
        median_s=176_400,
        rails=(0, 10_000, 0),
        lots={"x_saver_s1": (0, 320_000, 176_400), "x_saver_back": (10_000, 10_000, 3_600)},
    ),
    # A refund of 150,000 at t - 12 d 08:00 after 78 quiet days, spent by card after 7 h
    # (25,200 s, not under 6 h), 29 h and 36 h: 67,000 of it, under half.
    "a_sleeper": Hand(
        received=(0, 150_000, 150_000),
        sent=(0, 67_000, 67_000),
        total=150_000,
        within=(0, 0, 0, 20_000, 67_000, 67_000),
        matched=67_000,
        median_s=None,
        rails=(0, 20_000, 0),
        lots={"x_sleeper_refund": (20_000, 150_000, None)},
    ),
    # The mortgage at t - 30 d 09:00 comes before every lot of the span and matches nothing.
    # Three rents of 120,000; the plumber's 30,000 at t - 1 d 15:00 takes from the newest,
    # x_rent3_4 at t - 2 d 08:00, after 31 h.
    "a_landlord": Hand(
        received=(360_000, 360_000, 9 * 120_000),
        sent=(30_000, 250_000 + 30_000, 2 * 250_000 + 30_000),
        total=360_000,
        within=(0, 0, 0, 0, 30_000, 30_000),
        matched=30_000,
        median_s=None,
        rails=(0, 0, 0),
        lots={
            "z_rent1_4": (0, 120_000, None),
            "z_rent2_4": (0, 120_000, None),
            "x_rent3_4": (0, 120_000, None),
        },
    ),
    # 200,000 from the parent at t - 20 d 19:00: 190,000 to the school by ACH after 16 h
    # (57,600 s) and 6,000 at the cafe after 42 h; 30,000 at t - 6 d 19:00, of which 4,500 by
    # card after 17 h.
    "a_student": Hand(
        received=(30_000, 230_000, 260_000),
        sent=(4_500, 200_500, 200_500 + 25_000),
        total=230_000,
        within=(0, 0, 0, 190_000 + 4_500, 200_500, 200_500),
        matched=200_500,
        median_s=57_600,
        rails=(190_000, 4_500, 0),
        lots={"z_allowance_20": (190_000, 200_000, 57_600), "z_allowance_6": (4_500, 30_000, None)},
    ),
    # The power bill at t - 25 d matches nothing; the new payroll of 240,000 at t - 3 d 09:00
    # gives 8,000 by card after 3 h and 110,000 rent after 25 h: 118,000, just under half.
    "a_newjob": Hand(
        received=(240_000, 240_000, 240_000 + 3 * 220_000),
        sent=(118_000, 178_000, 178_000 + 2 * 110_000 + 60_000),
        total=240_000,
        within=(0, 0, 8_000, 8_000, 118_000, 118_000),
        matched=118_000,
        median_s=None,
        rails=(0, 8_000, 0),
        lots={"x_newco_3": (8_000, 240_000, None)},
    ),
    # Payroll L0 of 180,000 at t - 15 d 09:00: 20,000 from an ATM after 12 h, 40,000 by card
    # after 33 h. The victims' lot of one second, 100,000 + 50,000 at t - 6 d 14:00: 139,500
    # forwarded after 9 min (540 s), 7% kept. x_vic3's 200,000 at t - 4 d 20:00: 186,000
    # forwarded after 30 min, 7% kept. The median: 139,500 by 540 s, 325,500 by 1,800 s, past
    # half of 530,000.
    "a_mule": Hand(
        received=(350_000, 530_000, 530_000 + 2 * 180_000),
        sent=(325_500, 385_500, 385_500 + 2 * 160_000),
        total=530_000,
        within=(139_500, 325_500, 325_500, 345_500, 385_500, 385_500),
        matched=385_500,
        median_s=1_800,
        rails=(325_500, 0, 20_000),
        lots={
            "x_mule_pay15": (20_000, 180_000, None),
            "z_mule_v1": (139_500, 150_000, 540),
            "x_mule_v2": (139_500, 150_000, 540),
            "x_mule_v3": (186_000, 200_000, 1_800),
        },
    ),
    # Lots F1 139,500 (t - 6 d 14:09), G1 93,000 (t - 5 d 10:00), F2 186,000 (t - 4 d
    # 20:30), G2 46,500 (t - 3 d 16:00). The sweep of 450,000 at t - 2 d 12:00 takes G2 after
    # 20 h (72,000 s), F2 after 39 h 30 min (142,200 s), G1 after 74 h (266,400 s) and 124,500
    # of F1 after 93 h 51 min (337,860 s); the ATM's 10,000 at t - 1 d 02:00 takes F1 after
    # 107 h 51 min (388,260 s). The median: 46,500, then 232,500, exactly half of 465,000.
    "a_collector": Hand(
        received=(465_000, 465_000, 465_000),
        sent=(460_000, 460_000, 460_000),
        total=465_000,
        within=(0, 0, 0, 46_500, 46_500 + 186_000, 460_000),
        matched=460_000,
        median_s=142_200,
        rails=(46_500, 0, 0),
        lots={
            "z_mule_f1": (0, 139_500, 337_860),
            "z_collector_g2": (0, 93_000, 266_400),
            "z_mule_f2": (0, 186_000, 142_200),
            "z_collector_g3": (46_500, 46_500, 72_000),
        },
    ),
    # No payment at all: an old account, closed 200 days before t, that still exists.
    "a_newjob_old": Hand(
        received=(0, 0, 0),
        sent=(0, 0, 0),
        total=0,
        within=(0, 0, 0, 0, 0, 0),
        matched=0,
        median_s=None,
        rails=(0, 0, 0),
        lots={},
    ),
}


@pytest.fixture(scope="module")
def catalogue() -> ReferenceGraph:
    return fixtures.catalogue().graph()


@pytest.mark.parametrize("account", LOOK_ALIKES)
def test_each_look_alike_reads_as_worked_out_by_hand(
    catalogue: ReferenceGraph, account: str
) -> None:
    found = _flows(catalogue, [account])
    row, tracked = LOOK_ALIKES[account].rows(account)
    assert found.accounts == (row,)
    assert list(found.tracked_inflows) == tracked


def test_a_request_answers_each_account_as_it_answers_it_alone(
    catalogue: ReferenceGraph,
) -> None:
    accounts = list(LOOK_ALIKES)
    together = _flows(catalogue, accounts)
    alone = [_flows(catalogue, [account]) for account in sorted(accounts)]
    assert together.accounts == tuple(flow.accounts[0] for flow in alone)
    assert together.tracked_inflows == tuple(i for flow in alone for i in flow.tracked_inflows)


def _ties(w: WorldBuilder) -> WorldBuilder:
    """w with a_ties: lot A of 100,000 at t - 3 d 10:00; lot B of one second at t - 2 d
    10:00, 40,000 and 60,000 from two payers; a Zelle outflow of 10,000 in B's second; then,
    all in the second of t - 2 d 10:30, a card outflow of 100,000 (x_ties_o1) and two ACH
    outflows of 45,000 (x_ties_o2, x_ties_o3)."""
    w.customer("ties", fixtures.JOINED)
    for account in ("x_ties_a", "x_ties_b", "x_ties_shop", "x_ties_bank"):
        w.external(account, fixtures.OPENED)
    w.pay("x_ties_a1", "x_ties_a", "a_ties", T - 3 * D + 10 * H, 1_000.0, "ach")
    w.pay("x_ties_in1", "x_ties_a", "a_ties", T - 2 * D + 10 * H, 400.0, "ach")
    w.pay("x_ties_in2", "x_ties_b", "a_ties", T - 2 * D + 10 * H, 600.0, "ach")
    w.pay("z_ties_o0", "a_ties", "x_ties_bank", T - 2 * D + 10 * H, 100.0, "zelle")
    at = T - 2 * D + 10 * H + 30 * M
    w.pay("x_ties_o1", "a_ties", "x_ties_shop", at, 1_000.0, "card")
    w.pay("x_ties_o2", "a_ties", "x_ties_bank", at, 450.0, "ach")
    w.pay("x_ties_o3", "a_ties", "x_ties_bank", at, 450.0, "ach")
    return w


def test_a_lot_comes_before_its_seconds_outflows_which_go_by_rail_then_id() -> None:
    found = _flows(_ties(WorldBuilder()).graph(), ["a_ties"])
    # The Zelle outflow of B's second takes 10,000 of B at dwell 0. At 10:30 the two ACH
    # outflows go first, by the name of their rail, and take B's other 90,000 after 1,800 s;
    # the card outflow, whose id sorts first, comes last and takes A after 24 h 30 min (88,200
    # s, not under a day). Were ids to decide, the card would take B's newest cents within an
    # hour.
    row, tracked = Hand(
        received=(200_000, 200_000, 200_000),
        sent=(200_000, 200_000, 200_000),
        total=200_000,
        within=(10_000, 100_000, 100_000, 100_000, 200_000, 200_000),
        matched=200_000,
        median_s=1_800,
        rails=(100_000, 0, 0),
        lots={
            "x_ties_a1": (0, 100_000, 88_200),
            "x_ties_in1": (100_000, 100_000, 1_800),
            "x_ties_in2": (100_000, 100_000, 1_800),
        },
    ).rows("a_ties")
    assert found.accounts == (row,) and list(found.tracked_inflows) == tracked


def test_the_walk_reads_from_t_minus_30_days_to_just_before_t() -> None:
    w = WorldBuilder()
    w.customer("edge", fixtures.JOINED)
    w.external("x_edge_payer", fixtures.OPENED)
    w.external("x_edge_shop", fixtures.OPENED)
    for payment_id, ts, amount in (
        ("x_edge_in_30d", T - 30 * D, 1_000.0),
        ("x_edge_in_before_30d", T - 30 * D - 1, 500.0),
        ("x_edge_in_7d", T - 7 * D, 10.0),
        ("x_edge_in_90d", T - LOOKBACK_MS, 70.0),
        ("x_edge_in_before_90d", T - LOOKBACK_MS - 1, 80.0),
        ("x_edge_in_zero", T - D, 0.004),
        ("x_edge_in_t", T, 50.0),
    ):
        w.pay(payment_id, "x_edge_payer", "a_edge", ts, amount, "ach")
    for payment_id, ts, amount, rail in (
        ("x_edge_out", T - 30 * D + 10 * M, 300.0, "card"),
        ("x_edge_out_unknown", T - 7 * D + H, 2.0, "unknown"),
        ("x_edge_out_day", T - 6 * D, 5.0, "ach"),
        ("x_edge_out_t", T, 200.0, "card"),
    ):
        w.pay(payment_id, "a_edge", "x_edge_shop", ts, amount, rail)
    found = _flows(w.graph(), ["a_edge"])
    # The inflow at exactly t - 30 days is a lot; one a millisecond earlier, one at t and one
    # of 0.004 (0 cents) are not. The card outflow 10 minutes later takes 30,000 of it after
    # 600 s. The lot of 1,000 at exactly t - 7 days gives 200 on rail unknown after exactly an
    # hour (not under it), and 500 by ACH after exactly a day (not under it): of the 30,200
    # matched within a day, 30,000 left by card and 200, unknown, by no rail of the three.
    # The outflow at t is outside the lookback and matches nothing. The amounts count from
    # t - 90 days, the 0 cents included, to just before t.
    row, tracked = Hand(
        received=(1_000, 101_000, 101_000 + 50_000 + 7_000),
        sent=(700, 30_700, 30_700),
        total=101_000,
        within=(30_000, 30_000, 30_200, 30_200, 30_700, 30_700),
        matched=30_700,
        median_s=None,
        rails=(0, 30_000, 0),
        lots={"x_edge_in_30d": (30_000, 100_000, None), "x_edge_in_7d": (200, 1_000, 86_400)},
    ).rows("a_edge")
    assert found.accounts == (row,) and list(found.tracked_inflows) == tracked


def test_a_cut_list_drops_every_payment_at_or_before_the_cut_second() -> None:
    w = WorldBuilder()
    w.customer("busy", fixtures.JOINED)
    w.external("x_busy_payer", fixtures.OPENED)
    w.external("x_busy_shop", fixtures.OPENED)
    for k in range(300):
        w.pay(f"x_busy_o{k:03d}", "a_busy", "x_busy_shop", T - 10 * D + k * M, 10.0, "card")
    for payment_id, ts, amount in (
        ("x_busy_a", T - 20 * D + 9 * H, 5_000.0),
        ("x_busy_c", T - 10 * D + 43 * M, 100.0),
        ("x_busy_d", T - 10 * D + 44 * M + 500, 50.0),
        ("x_busy_b", T - 10 * D + 100 * M + 30_000, 1_000.0),
    ):
        w.pay(payment_id, "x_busy_payer", "a_busy", ts, amount, "ach")
    found = _flows(w.graph(), ["a_busy"])
    # 300 outflows of 1,000, one a minute from t - 10 days (k = 0 to 299). The list keeps the
    # newest 256, so the newest it leaves out is k = 43 and b is its second. Every payment at
    # or before b is dropped: outflows 0 to 43, lot A of t - 20 days and lot C of b itself.
    # Lot D of 5,000, in k = 44's second, comes before that outflow and is taken by outflows
    # 44 to 48 after 0, 60, 120, 180 and 240 s, half by 120 s. Lot B of 100,000 at minute 100
    # and 30 s is taken by outflows 101 to 200, outflow k after (k - 100) x 60 - 30 s: 15
    # under 15 minutes (to k = 115), 60 under an hour (to k = 160), all under 6 hours; half of
    # B by k = 150 (2,970 s). Of R = 105,000, half is reached with D's 5,000 and 48 pieces of
    # B, at k = 148 (2,850 s).
    row, tracked = Hand(
        received=(0, 615_000, 615_000),
        sent=(0, 300_000, 300_000),
        total=105_000,
        within=(5_000 + 15_000, 5_000 + 60_000, 105_000, 105_000, 105_000, 105_000),
        matched=105_000,
        median_s=2_850,
        rails=(0, 105_000, 0),
        lots={"x_busy_d": (5_000, 5_000, 120), "x_busy_b": (100_000, 100_000, 2_970)},
    ).rows("a_busy")
    assert found.accounts == (row,) and list(found.tracked_inflows) == tracked


# The walk against full-history last-in, first-out matching.


def _full_history(
    received: Sequence[Payment], sent: Sequence[Payment]
) -> dict[int, dict[str, int]]:
    """Each second's inflows matched last in, first out over every payment, uncapped, with a
    stack of payments rather than lots: per second of inflows, the cents each outflow took.
    The inflows of a second go before its outflows, and the outflows of a second by rail, then
    id, as docs/features.md orders them."""
    events = sorted(
        [(second(p.ts), 0, "", p.id, p) for p in received if p.cents > 0]
        + [(second(p.ts), 1, p.rail, p.id, p) for p in sent],
        key=lambda event: event[:4],
    )
    stack: list[tuple[Payment, list[int]]] = []
    taken: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for _, kind, _, _, payment in events:
        if kind == 0:
            stack.append((payment, [payment.cents]))
            continue
        need = payment.cents
        while need > 0 and stack:
            inflow, left = stack[-1]
            take = min(need, left[0])
            taken[second(inflow.ts)][payment.id] += take
            left[0] -= take
            need -= take
            if left[0] == 0:
                stack.pop()
    return taken


def _payments(rng: random.Random, count: int, first_s: int, span_s: int) -> list[Payment]:
    """count payments at random milliseconds of the span_s seconds from first_s, of random
    cents (0 among them) and rails, with random ids unrelated to their times."""
    found: list[Payment] = []
    for index in range(count):
        cents = rng.choice((0, 1, 7, 100, 999, 50_000, rng.randrange(1, 200_000)))
        ts = (first_s + rng.randrange(span_s)) * 1_000 + rng.randrange(1_000)
        payment_id = f"{rng.getrandbits(40):010x}{index}"
        rail = rng.choice(RAILS)
        found.append(Payment(payment_id, "", cents / 100, cents, rail, ts, "", "", None, None))
    return found


def test_the_walk_matches_full_history_last_in_first_out_on_2000_random_accounts() -> None:
    rng = random.Random(20261018)
    since_s = (T - 30 * D) // 1_000
    cut_accounts = 0
    for _ in range(2_000):
        cap = rng.choice((1, 2, 3, 5, 8, 16, RECENT))
        most = 300 if cap == RECENT else 25
        span_s = rng.choice((3, 40, 5_000, 30 * 86_400))
        received = _payments(rng, rng.randrange(most), since_s, span_s)
        sent = _payments(rng, rng.randrange(most), since_s, span_s)
        # Payments of the day before the 30 days, which the walk never reads and full history
        # does.
        older_in = _payments(rng, rng.randrange(4), since_s - 86_400, 86_400)
        older_out = _payments(rng, rng.randrange(4), since_s - 86_400, 86_400)
        found_walk = walk(received, sent, cap)
        full = _full_history([*older_in, *received], [*older_out, *sent])
        positive: dict[int, int] = defaultdict(int)
        for payment in received:
            positive[second(payment.ts)] += payment.cents
        # The span: after b, the latest second of the newest payment a cut list left out.
        lists = ([p.ts for p in received if p.cents > 0], [p.ts for p in sent])
        left_out = [sorted(times)[-cap - 1] // 1_000 for times in lists if len(times) > cap]
        b = max(left_out, default=None)
        kept = {lot.second for lot in found_walk.lots}
        assert kept == {s for s, c in positive.items() if c > 0 and (b is None or s > b)}
        cut_accounts += b is not None and kept != {s for s, c in positive.items() if c > 0}
        by_lot: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for piece in found_walk.pieces:
            by_lot[found_walk.lots[piece.lot].second][piece.outflow] += piece.cents
        for lot in found_walk.lots:
            # Every kept lot holds all its second's inflows and is spent as full history
            # spends it, outflow by outflow.
            assert lot.cents == positive[lot.second]
            assert dict(by_lot[lot.second]) == dict(full.get(lot.second, {}))
    # Enough of the accounts had a cut list to test the span.
    assert cut_accounts > 300


# Refusals.


@pytest.mark.parametrize(
    ("params", "status"),
    [
        ({"t_ms": T + D, "account_ids": ["a_mule"]}, "invalid_parameters"),
        ({"t_ms": T, "account_ids": []}, "invalid_parameters"),
        ({"t_ms": T, "account_ids": ["a_mule", "a_mule"]}, "duplicate_account"),
        ({"t_ms": T, "account_ids": ["a_mule", "a_nobody"]}, "unknown_account"),
        ({"t_ms": T, "account_ids": ["a_mule", "a_late"]}, "invisible_account"),
    ],
    ids=["no Monday", "no account", "an account twice", "an unknown account", "opened at t"],
)
def test_the_query_refuses_a_request_it_cannot_answer(
    params: dict[str, object], status: str
) -> None:
    w = fixtures.catalogue()
    w.customer("late", T)
    rows = w.graph().run(READ_MONEY_FLOW, params)
    assert rows == [{"status": status}]
    with pytest.raises(QueryRefused, match=status):
        read_money_flow.parse(rows, ["a_mule"])


# The leak tests over B1.


def _printed(graph: ReferenceGraph) -> str:
    """The answer for every account that exists at t, as the bytes TigerGraph would send."""
    return json.dumps(graph.run(READ_MONEY_FLOW, read_money_flow.params(T, _existing(graph))))


def test_erasing_every_label_column_changes_no_answer() -> None:
    tables = fixtures.catalogue().tables()
    erased = fixtures.erase_labels(tables)
    assert not erased["Account"].equals(tables["Account"])
    assert _printed(ReferenceGraph(erased)) == _printed(ReferenceGraph(tables))


def test_moving_dates_of_record_before_the_lookback_changes_no_answer() -> None:
    tables = fixtures.catalogue().tables()
    moved = fixtures.move_dates_of_record(tables, T, LOOKBACK_MS)
    for name in ("Party", "Account", "Party_Owns_Account"):
        assert not moved[name].equals(tables[name])
    assert _printed(ReferenceGraph(moved)) == _printed(ReferenceGraph(tables))


def test_swapping_the_ids_of_same_second_payments_changes_no_value() -> None:
    tables = _ties(fixtures.catalogue()).tables()
    swapped, original = fixtures.swap_same_second_ids(tables)
    assert any(new != old for new, old in original.items())
    before, after = ReferenceGraph(tables), ReferenceGraph(swapped)
    accounts = _existing(before)
    found = _flows(after, accounts)
    renamed = tuple(
        sorted(
            (
                TrackedInflow(
                    i.account_id, original[i.payment_id], i.left_24h, i.dwell_present, i.dwell_ms
                )
                for i in found.tracked_inflows
            ),
            key=lambda inflow: (inflow.account_id, inflow.payment_id),
        )
    )
    expected = _flows(before, accounts)
    assert found.accounts == expected.accounts and renamed == expected.tracked_inflows


def test_a_self_payment_is_read_nowhere() -> None:
    w = fixtures.catalogue()
    before = _printed(w.graph())
    w.pay("x_mule_self", "a_mule", "a_mule", T - 6 * D + 14 * H + M, 900.0, "internal")
    assert _printed(w.graph()) == before
