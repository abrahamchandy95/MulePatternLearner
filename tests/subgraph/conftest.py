"""What the cache, fetch and batch tests share: a reference world, a parity record of the
test's own, and the rows of a hand-built batch of two examples (hand_rows, below).

The world is testing.fixtures.batch_world: the catalogue of look-alikes with a household, a
ring, the owner's Account Z and a shop that is an account hub at T.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mule_pattern_learner.subgraph.build import Batch, ExampleRows, Leaves, build
from mule_pattern_learner.subgraph.cache import INDEX_FILE
from mule_pattern_learner.testing import fixtures
from mule_pattern_learner.testing.reference import ReferenceGraph
from mule_pattern_learner.testing.world import DAY_MS, HOUR_MS, T
from mule_pattern_learner.tigergraph.queries.read_activity_change import ActivityChange
from mule_pattern_learner.tigergraph.queries.read_community import Community
from mule_pattern_learner.tigergraph.queries.read_counterparties import Counterparties
from mule_pattern_learner.tigergraph.queries.read_hubs import HubAccount, Hubs, Leaf
from mule_pattern_learner.tigergraph.queries.read_money_flow import MoneyFlow, TrackedInflow
from mule_pattern_learner.tigergraph.queries.read_payment_events import (
    EventAccount,
    OwnedBy,
    PaymentEvent,
    PaymentGraph,
)
from mule_pattern_learner.tigergraph.queries.read_shared_identity import (
    CustomerRow,
    Enrolled,
    IdentityAccount,
    IdentityGraph,
    NodeRow,
    OwnerEnrolled,
    Seen,
)


@pytest.fixture(scope="session")
def graph() -> ReferenceGraph:
    """The reference graph of the world of this module's docstring."""
    return fixtures.batch_world().graph()


@pytest.fixture
def parity(tmp_path: Path) -> Path:
    """A parity record of the test's own holding every query's current version."""
    record = tmp_path / "parity.toml"
    fixtures.write_parity_record(record)
    return record


@pytest.fixture
def index_reads(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """The part indexes the cache reads from now on, in order: a cache holds a Monday's indexes
    in memory, so each read is one that was not held."""
    reads: list[Path] = []
    read_table = pq.read_table

    def counting(source: Path, *args: Any, **keywords: Any) -> pa.Table:
        if Path(source).name == INDEX_FILE:
            reads.append(Path(source))
        return read_table(source, *args, **keywords)

    monkeypatch.setattr(pq, "read_table", counting)
    return reads


# A hand-built batch of two examples, which the batch, degree and switch tests read.
#
# Example 0, at T, scores a_s1, whose customer p_c1 also owns a_own. a_s1 receives p1 from
# x_payer, another bank's account, and pays a_hub, an account hub (p2), and a_own (p4); a_two,
# a hop-2 account, paid x_payer (p3). p_c1 is enrolled on the device d_home and the IP i_home;
# p_co, a co-user, is enrolled on d_home too and brings its account a_co; a_two is a co-user's
# account too, seen on i_home, so it is reached by a payment and by an IP. a_s1's payments were
# seen on d_home and on d_crowd, a crowd.
#
# Example 1, at T + 1 week, scores a_s2, itself an account hub that Monday, whose customer is
# p_c2. x_payer pays it (p5), and it pays a_three (p6); a_s2's payments were seen on i_crowd, an
# IP hub. x_payer is in both examples, each with its own Monday's rows.
#
# Each account's B1 to B3 rows are numbered by k, so that every column of every row differs.
HAND_T1 = T + 7 * DAY_MS
HAND_K = {
    (T, "a_s1"): 1,
    (T, "x_payer"): 2,
    (T, "a_two"): 3,
    (T, "a_own"): 4,
    (T, "a_co"): 5,
    (HAND_T1, "a_s2"): 6,
    (HAND_T1, "a_three"): 7,
    (HAND_T1, "x_payer"): 8,
}
MIN = 60_000


def hand_flow(account: str, k: int) -> MoneyFlow:
    return MoneyFlow(
        account_id=account,
        in_amount_7d=1000 * k + 1,
        in_amount_30d=1000 * k + 2,
        in_amount_90d=1000 * k + 3,
        out_amount_7d=1000 * k + 4,
        out_amount_30d=1000 * k + 5,
        out_amount_90d=1000 * k + 6,
        flow_present=True,
        left_15m=0.1 + k / 1000,
        left_1h=0.2 + k / 1000,
        left_6h=0.3 + k / 1000,
        left_1d=0.4 + k / 1000,
        left_3d=0.5 + k / 1000,
        left_7d=0.6 + k / 1000,
        held_share=0.35 + k / 1000,
        median_dwell_present=True,
        median_dwell_ms=k * MIN,
        left_by_transfer=0.11 + k / 1000,
        left_by_card=0.12 + k / 1000,
        left_by_cash=0.13 + k / 1000,
        scan_steps=99,
    )


def hand_counterparties(account: str, k: int) -> Counterparties:
    return Counterparties(
        account_id=account,
        payers_30d=10 * k + 1,
        payees_30d=10 * k + 2,
        new_internal_payers_7d=10 * k + 3,
        new_internal_payers_30d=10 * k + 4,
        new_external_payers_7d=10 * k + 5,
        new_external_payers_30d=10 * k + 6,
        new_payees_7d=10 * k + 7,
        new_payees_30d=10 * k + 8,
        new_payer_share_30d=0.7 + k / 1000,
        max_new_payers_24h=10 * k + 9,
        top_payee_share_30d=0.8 + k / 1000,
        hub_payee_share_30d=0.9 + k / 1000,
    )


def hand_activity(account: str, k: int) -> ActivityChange:
    return ActivityChange(
        account_id=account,
        in_7d=100 * k + 11,
        in_30d=100 * k + 12,
        in_90d=100 * k + 13,
        out_7d=100 * k + 14,
        out_30d=100 * k + 15,
        out_90d=100 * k + 16,
        burst_in_1h=100 * k + 17,
        burst_in_24h=100 * k + 18,
        burst_out_1h=100 * k + 19,
        burst_out_24h=100 * k + 20,
        shortest_in_gap_present=True,
        shortest_in_gap_ms=k * 1000,
        since_last_in_present=True,
        since_last_in_ms=k * HOUR_MS,
        since_last_out_present=False,
        since_last_out_ms=0,
    )


def _payment(
    payment_id: str,
    sender: str,
    receiver: str,
    amount: int,
    rail: str,
    age_ms: int,
    prior: int = 0,
    pair: tuple[int, int] = (0, 0),
    sender_gap: int = 0,
    receiver_gap: int = 0,
) -> PaymentEvent:
    return PaymentEvent(
        payment_id,
        sender,
        receiver,
        amount,
        rail,
        age_ms,
        prior,
        prior > 0,
        pair[0],
        prior > 0,
        pair[1],
        sender_gap > 0,
        sender_gap,
        receiver_gap > 0,
        receiver_gap,
    )


def hand_rows() -> list[ExampleRows]:
    """The rows of the two examples of the hand-built batch."""
    leaves0 = Leaves.of(Hubs((HubAccount("a_hub", 250),), (Leaf("d_crowd", 12, True, False),), ()))
    payments0 = PaymentGraph(
        "a_s1",
        tuple(
            EventAccount(account, account.startswith("x_"))
            for account in ("a_hub", "a_own", "a_s1", "a_two", "x_payer")
        ),
        (
            _payment(
                "p1",
                "x_payer",
                "a_s1",
                50_000,
                "zelle",
                3 * DAY_MS,
                0,
                (0, 0),
                2 * HOUR_MS,
                5 * DAY_MS,
            ),
            _payment(
                "p2",
                "a_s1",
                "a_hub",
                1_999,
                "card",
                2 * DAY_MS,
                2,
                (30 * DAY_MS, 60 * DAY_MS),
                DAY_MS,
            ),
            _payment(
                "p3", "a_two", "x_payer", 70_000, "ach", 4 * DAY_MS, 1, (7 * DAY_MS, 7 * DAY_MS)
            ),
            _payment(
                "p4",
                "a_s1",
                "a_own",
                47_500,
                "internal",
                3 * DAY_MS - 10 * MIN,
                0,
                (0, 0),
                10 * MIN,
                6 * DAY_MS,
            ),
        ),
        (OwnedBy("a_own", "p_c1"), OwnedBy("a_s1", "p_c1")),
    )
    identity0 = IdentityGraph(
        "a_s1",
        (CustomerRow("p_c1", 2, 3, 1, 1, 1, 1, 1, 1), CustomerRow("p_co", 1, 1, 2, 1, 1, 1, 0, 0)),
        (
            NodeRow("d_crowd", True, 0, 0, 0, 0, 0, 0, False),
            NodeRow("d_home", False, 2, 3, 1, 2, 2, 5, False),
        ),
        (NodeRow("i_home", False, 1, 2, 0, 1, 1, 4, False),),
        (IdentityAccount("a_co", False), IdentityAccount("a_two", False)),
        (
            Enrolled("p_c1", "d_home", True, 20 * DAY_MS, True, 3 * DAY_MS),
            Enrolled("p_c1", "i_home", False, 0, True, 3 * DAY_MS),
            Enrolled("p_co", "d_home", True, 5 * DAY_MS, False, 0),
        ),
        (
            OwnerEnrolled("a_co", "d_home", True, 5 * DAY_MS, False, 0),
            OwnerEnrolled("a_own", "d_home", True, 20 * DAY_MS, True, 3 * DAY_MS),
            OwnerEnrolled("a_s1", "d_home", True, 20 * DAY_MS, True, 3 * DAY_MS),
            OwnerEnrolled("a_s1", "i_home", False, 0, True, 3 * DAY_MS),
        ),
        (
            Seen("a_s1", "d_crowd", 2, False, 0, 3 * DAY_MS),
            Seen("a_s1", "d_home", 1, True, 3 * DAY_MS, 3 * DAY_MS),
            Seen("a_two", "i_home", 4, True, 40 * DAY_MS, 4 * DAY_MS),
        ),
    )
    community0 = Community("a_s1", 3, False, 1, 0.75, 0.5, 47_500, 0.25, 4)
    inflows0 = {
        "p1": TrackedInflow("a_s1", "p1", 0.95, True, 10 * MIN),
        "p3": TrackedInflow("x_payer", "p3", 0.0, False, 0),
    }
    leaves1 = Leaves.of(Hubs((HubAccount("a_s2", 300),), (), (Leaf("i_crowd", 220, False, True),)))
    payments1 = PaymentGraph(
        "a_s2",
        (
            EventAccount("a_s2", False),
            EventAccount("a_three", False),
            EventAccount("x_payer", True),
        ),
        (
            _payment("p5", "x_payer", "a_s2", 1_250, "check", DAY_MS),
            _payment(
                "p6", "a_s2", "a_three", 900, "unknown", 6 * HOUR_MS, 0, (0, 0), 0, 2 * DAY_MS
            ),
        ),
        (OwnedBy("a_s2", "p_c2"),),
    )
    identity1 = IdentityGraph(
        "a_s2",
        (CustomerRow("p_c2", 1, 0, 0, 0, 0, 0, 0, 0),),
        (),
        (NodeRow("i_crowd", True, 0, 0, 0, 0, 0, 0, False),),
        (),
        (),
        (),
        (Seen("a_s2", "i_crowd", 1, False, 0, DAY_MS),),
    )
    community1 = Community("a_s2", 1, False, 0, 0.0, 0.0, 0, 0.0, 7)
    # a_s2 is a hub, so B1 tracks p5 for it but its edge reads none; a_three tracks p6.
    inflows1 = {
        "p5": TrackedInflow("a_s2", "p5", 1.0, True, MIN),
        "p6": TrackedInflow("a_three", "p6", 0.5, True, 2 * HOUR_MS),
    }
    found: list[ExampleRows] = []
    for t_ms, leaves, payments, identity, community, inflows in (
        (T, leaves0, payments0, identity0, community0, inflows0),
        (HAND_T1, leaves1, payments1, identity1, community1, inflows1),
    ):
        ks = {account: k for (t, account), k in HAND_K.items() if t == t_ms}
        found.append(
            ExampleRows(
                t_ms,
                leaves,
                payments,
                identity,
                community,
                {account: hand_flow(account, k) for account, k in ks.items()},
                inflows,
                {account: hand_counterparties(account, k) for account, k in ks.items()},
                {account: hand_activity(account, k) for account, k in ks.items()},
            )
        )
    return found


@pytest.fixture
def hand() -> list[ExampleRows]:
    """The rows of the hand-built batch's two examples."""
    return hand_rows()


@pytest.fixture
def hand_batch(hand: list[ExampleRows]) -> Batch:
    """The hand-built training batch: a_s1 a positive weighing 2.5, a_s2 unlabelled weighing
    0.75."""
    return build(hand, "w", labels=[1, 0], weights=[2.5, 0.75])
