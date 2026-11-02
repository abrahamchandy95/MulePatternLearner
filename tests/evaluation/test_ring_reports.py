"""The two ring reports on a reference world, with the values worked out by hand (docs/design.md,
"Comparisons" and "The payment round"): the share of positives holding a ring node, and the share
of positives whose community block differs from that of a customer who shares nothing, by sharing
and by payments, by whether they hold a ring node."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mule_pattern_learner.evaluation.answer_key import AnswerKey
from mule_pattern_learner.evaluation.rankings import evaluation_frame
from mule_pattern_learner.evaluation.ring_reports import COMMUNITY, ENDPOINTS, ring_reports
from mule_pattern_learner.subgraph.cache import Cache
from mule_pattern_learner.subgraph.fetch import fetch
from mule_pattern_learner.testing import fixtures
from mule_pattern_learner.testing.calendar import Spy
from mule_pattern_learner.testing.world import DAY_MS, Mule, T, WorldBuilder
from mule_pattern_learner.tigergraph.queries.read_answer_key import AccountTruth

D = DAY_MS
ACCOUNTS = ["a_r1", "a_r2", "a_b1", "a_b2", "a_e1"]
KEY = AnswerKey(
    (
        AccountTruth("a_r1", 1, 7, T - 5 * D, T - D, 0, "deposit", True),
        AccountTruth("a_r2", 1, 7, T - 5 * D, T - D, 0, "deposit", True),
        AccountTruth("a_b1", 1, -1, T - 5 * D, T - D, 0, "deposit", True),
        AccountTruth("a_b2", 1, 8, T - 5 * D, T - D, 0, "deposit", True),
        AccountTruth("a_e1", 1, 7, T + 2 * D, T + 9 * D, 0, "deposit", True),
    )
)


def _world() -> WorldBuilder:
    """At the time T:

    - a_r1 and a_r2, customers r1 and r2 of ring 7, both active; both are enrolled on the device
      d_r since 30 days before t, so d_r is a ring node (two customers reach it) and both accounts
      hold it. Each community has the other customer as a member, through d_r.
    - a_b1, a ringless mule of customer b1, active, paying 100.00 by ACH to a_c, an ordinary
      customer, 2 days before t. Its community has c as a member and money moving between the
      two, so it differs. It holds no ring node.
    - a_b2, of ring 8, active and alone on the device d_solo, paying 50.00 to an external
      account only. No other customer reaches d_solo, and its community is a customer who shares
      nothing, so it does not differ.
    - a_e1, of ring 7 at its early-warning time (active from 2 days after t), enrolled on d_r.
      It is no positive, so it counts in neither table, and no ring node, since only active
      mules make ring nodes.
    """
    w = WorldBuilder()
    for name in ("r1", "r2"):
        w.customer(name, fixtures.JOINED, mule=Mule(T - 5 * D, T - D, ring=7))
        w.device(f"p_{name}", "d_r", T - 30 * D)
    w.customer("b1", fixtures.JOINED, mule=Mule(T - 5 * D, T - D))
    w.customer("c", fixtures.JOINED)
    w.pay("pay_b1_c", "a_b1", "a_c", T - 2 * D, 100.0, "ach")
    w.customer("b2", fixtures.JOINED, mule=Mule(T - 5 * D, T - D, ring=8))
    w.device("p_b2", "d_solo", T - 30 * D)
    w.external("x_b2", fixtures.OPENED)
    w.pay("pay_b2_x", "a_b2", "x_b2", T - 2 * D, 50.0, "ach")
    w.customer("e1", fixtures.JOINED, mule=Mule(T + 2 * D, T + 9 * D, ring=7))
    w.device("p_e1", "d_r", T - 30 * D)
    return w


def _frame(accounts: list[str]) -> pd.DataFrame:
    scores = pd.DataFrame({"t_ms": [T] * len(accounts), "account_id": accounts, "logit": 0.0})
    return evaluation_frame(
        scores, scores.loc[:, ["t_ms", "account_id"]].assign(weight=1.0), KEY, frozenset(), (T,)
    )


def test_the_ring_reports_of_a_reference_world(parity: Path, tmp_path: Path) -> None:
    cache = Cache(tmp_path / "cache")
    fetch(Spy(_world().graph()), cache, {T: ACCOUNTS}, parity=parity)
    tables = ring_reports(cache, _frame(ACCOUNTS))
    # Four positives: a_r1 and a_r2 hold d_r, and a_b1 and a_b2 hold no ring node.
    assert tables[ENDPOINTS].to_dict("records") == [
        {"positives": 4, "with_ring_node": 2, "share": 0.5}
    ]
    # By sharing, a_r1 and a_r2 differ (a shared node, d_r) and a_b1 and a_b2 do not (no shared
    # node). By payments, a_r1 and a_r2 differ (members r1, r2 and e1 through d_r), a_b1 differs
    # (member c, money moving between the two) and a_b2 does not.
    assert tables[COMMUNITY].to_dict("records") == [
        {
            "holds_ring_node": True,
            "positives": 2,
            "differs_by_sharing": 2,
            "share_by_sharing": 1.0,
            "differs_by_payments": 2,
            "share_by_payments": 1.0,
        },
        {
            "holds_ring_node": False,
            "positives": 2,
            "differs_by_sharing": 0,
            "share_by_sharing": 0.0,
            "differs_by_payments": 1,
            "share_by_payments": 0.5,
        },
    ]


def test_the_ring_reports_with_no_positive_have_no_share(parity: Path, tmp_path: Path) -> None:
    cache = Cache(tmp_path / "cache")
    fetch(Spy(_world().graph()), cache, {T: ACCOUNTS}, parity=parity)
    # a_e1 alone is at its early-warning time, so there is no positive to share anything.
    tables = ring_reports(cache, _frame(["a_e1"]))
    assert tables[ENDPOINTS]["positives"].tolist() == [0]
    assert np.isnan(tables[ENDPOINTS]["share"].iloc[0])
    assert tables[COMMUNITY]["positives"].tolist() == [0, 0]
