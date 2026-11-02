"""The groups of true mules, each ranked against the non-mules with the other groups removed, and
the hop distance from the nearest identified mule inside a mule's own subgraph (docs/design.md,
"Groups")."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mule_pattern_learner.evaluation.answer_key import AnswerKey
from mule_pattern_learner.evaluation.groups import (
    GROUPS,
    group_metrics,
    group_names,
    group_rows,
    quiet_rows,
    surfaced_rings,
)
from mule_pattern_learner.evaluation.hops import hop_distance, hop_distances
from mule_pattern_learner.evaluation.rankings import evaluation_frame
from mule_pattern_learner.subgraph.build import read_rows
from mule_pattern_learner.subgraph.cache import Cache
from mule_pattern_learner.subgraph.fetch import fetch
from mule_pattern_learner.testing.calendar import Spy
from mule_pattern_learner.testing.reference import ReferenceGraph
from mule_pattern_learner.testing.world import DAY_MS, T
from mule_pattern_learner.tigergraph.queries.read_answer_key import AccountTruth

D, W = DAY_MS, 7 * DAY_MS
# The world's one scoring time, which the status of each account is read at.
TIMES = (T,)
OLD = T - 400 * D
# Ring 3 surfaced through a; ring 4 never did. d is a ringless drop whose account opened 10 days
# before T; e is at its early-warning time, T.
KEY = AnswerKey(
    (
        AccountTruth("a", 1, 3, T - D, T + D, OLD, "deposit", True),
        AccountTruth("b", 1, 3, T - D, T + D, OLD, "deposit", True),
        AccountTruth("c", 1, 4, T - D, T + D, OLD, "deposit", True),
        AccountTruth("d", 1, -1, T - D, T + D, T - 10 * D, "deposit", True),
        AccountTruth("e", 1, 4, T + 2 * D, T + 3 * D, OLD, "deposit", True),
        *(AccountTruth(f"n{i}", 0, -1, 0, 0, OLD, "deposit", True) for i in range(4)),
    )
)
LOGITS = {"a": 0.5, "b": 0.05, "c": 0.95, "d": 0.6, "e": 0.9, "n0": 0.1, "n1": 0.2}


def _frame() -> pd.DataFrame:
    accounts = [truth.account_id for truth in KEY.truths]
    scores = pd.DataFrame(
        {"t_ms": [T] * 9, "account_id": accounts, "logit": [LOGITS.get(a, 0.3) for a in accounts]}
    )
    # Each non-mule stands for 100 accounts: 401 or more in every ranking, so the top 1% (4.01)
    # takes a mule of weight 1 at the top whole.
    keys = scores.loc[:, ["t_ms", "account_id"]].assign(
        weight=[100.0 if account[0] == "n" else 1.0 for account in accounts]
    )
    return evaluation_frame(scores, keys, KEY, frozenset({"a"}), TIMES)


def test_each_group_holds_its_rows() -> None:
    frame = _frame()
    assert surfaced_rings(KEY, frozenset({"a"})) == {3}
    hops = {(T, "a"): 2, (T, "b"): 1, (T, "c"): None, (T, "d"): 2}
    rows = group_rows(frame, {3}, {(T, "d"), (T, "e")}, hops)
    members = {name: sorted(frame.loc[rows[name], "account_id"]) for name in GROUPS}
    assert members == {
        "hidden_ring": ["c"],
        "surfaced_unidentified": ["b"],
        "identified": ["a"],
        "no_ring": ["d"],
        "new_account": ["d"],
        # The early-warning time counts in this group alone.
        "no_payment": ["d", "e"],
        "hop_1": ["b"],
        "hop_2": ["a", "d"],
        "hop_beyond": ["c"],
    }


def test_each_account_is_named_by_the_groups_that_hold_its_rows() -> None:
    frame = _frame()
    hops = {(T, "a"): 2, (T, "b"): 1, (T, "c"): None, (T, "d"): 2}
    rows = group_rows(frame, {3}, {(T, "d"), (T, "e")}, hops)
    # In the order of GROUPS; a non-mule is in none, so it is no key.
    assert group_names(frame, rows) == {
        "a": "identified,hop_2",
        "b": "surfaced_unidentified,hop_1",
        "c": "hidden_ring,hop_beyond",
        "d": "no_ring,new_account,no_payment,hop_2",
        "e": "no_payment",
    }


def test_each_group_is_ranked_with_the_other_mules_removed() -> None:
    frame = _frame()
    rows = group_rows(frame, {3}, {(T, "d"), (T, "e")}, {})
    table = group_metrics(frame, rows).set_index(["group", "metric"])["value"]
    # c tops its ranking and b is under every non-mule. a at 0.5 is above every non-mule (0.1 to
    # 0.3), and the mules above it, c at 0.95, e at 0.9 and d at 0.6, are removed from its
    # ranking: it tops it.
    assert table[("hidden_ring", "recall_at_1pct")] == 1
    assert table[("surfaced_unidentified", "recall_at_5pct")] == 0
    assert table[("identified", "recall_at_1pct")] == 1
    assert (
        table[("identified", "roc_auc")] == 1 and table[("surfaced_unidentified", "roc_auc")] == 0
    )
    # e alone of no_payment is at its early-warning time: d at 0.6 and e at 0.9 both top it.
    assert table[("no_payment", "mules")] == 2 and table[("no_payment", "recall_at_5pct")] == 1
    assert table[("hop_1", "account_weeks")] == 0 and ("hop_1", "roc_auc") not in table.index
    assert table[("hidden_ring", "ever_flagged_at_1pct")] == 1
    assert table[("no_payment", "enrolment_ceiling_device")] == 0.72


def test_hop_distances_in_a_mules_own_subgraph(
    graph: ReferenceGraph, parity: Path, tmp_path: Path
) -> None:
    cache = Cache(tmp_path / "cache")
    fetch(Spy(graph), cache, {T: ["a_hr2", "a_hr3", "a_o00"], T + W: ["a_late"]}, parity=parity)
    hr2, hr3, ordinary = read_rows(cache, [(T, "a_hr2"), (T, "a_hr3"), (T, "a_o00")])
    # a_hr1 paid a_hr2: one edge. a_hr3 was paid by a_hr2, which a_hr1 paid: two edges, and
    # two too through the ring's device, which a_hr1's account is enrolled on.
    assert hop_distance(hr2, {"a_hr1"}) == 1 and hop_distance(hr3, {"a_hr1"}) == 2
    # The scored account is never its own nearest identified mule.
    assert hop_distance(hr2, {"a_hr2"}) is None and hop_distance(ordinary, {"a_hr1"}) is None
    # a_late has no payment: only its customer's enrolment on the ring's device, which ring 11's
    # accounts are owner-enrolled on, puts a_hr1 two edges away (the owner's Account Z).
    (late,) = read_rows(cache, [(T + W, "a_late")])
    assert not late.payments.payments and hop_distance(late, {"a_hr1"}) == 2
    found = hop_distances(cache, [(T, "a_hr2"), (T, "a_o00")], {"a_hr1"})
    assert found == {(T, "a_hr2"): 1, (T, "a_o00"): None}


def test_quiet_rows_have_no_payment_in_the_lookback(
    graph: ReferenceGraph, parity: Path, tmp_path: Path
) -> None:
    cache = Cache(tmp_path / "cache")
    fetch(Spy(graph), cache, {T: ["a_hr2"], T + W: ["a_late"]}, parity=parity)
    assert quiet_rows(cache, [(T, "a_hr2"), (T + W, "a_late")]) == {(T + W, "a_late")}
