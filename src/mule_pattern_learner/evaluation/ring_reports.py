"""The two reports on a ring's devices and IPs that mule evaluate writes beside the device and IP
ablation and the community block (docs/design.md, "Comparisons" and "The payment round"). They
read the positives of a run's test world, the active true mules of its frame, and their rows from
the cache, and nothing else. They are evaluation outputs, never features.

- ring_endpoints: the positives, and how many hold any ring node at their t: a device or IP
  that the own edges of active true mules of their ring, at two or more customers, reach
  (development_checks.ring_nodes).
- ring_community: the positives by whether they hold a ring node, and how many of each have a
  community block that differs from that of a customer who shares nothing, counted twice: by
  sharing (shared nodes, density, new link share) and by payments (members, internal amount,
  isolation).
"""

from __future__ import annotations

import math

import pandas as pd

from ..subgraph.build import read_rows
from ..subgraph.cache import Cache
from ..tigergraph.queries.read_community import Community
from .development_checks import ActiveMule, customer_of, own_nodes, ring_nodes
from .statuses import ACTIVE

ENDPOINTS = "ring_endpoints"
COMMUNITY = "ring_community"


def differs_by_sharing(block: Community) -> bool:
    """Whether a community block differs by sharing from that of a customer who shares nothing,
    whose shared nodes, density and new link share read 0, 0, 0."""
    return block.shared_nodes != 0 or block.density != 0 or block.new_link_share_30d != 0


def differs_by_payments(block: Community) -> bool:
    """Whether a community block differs by payments from that of a customer who shares nothing,
    whose members, internal amount and isolation read 1, 0, 0: its owner alone, and no money
    moving between members."""
    return block.members != 1 or block.internal_amount_30d != 0 or block.isolation != 0


def ring_reports(cache: Cache, frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The tables ENDPOINTS and COMMUNITY of a run's test world. frame holds every scored
    account-week with its status and ring (evaluation.rankings.evaluation_frame), and the cache
    the rows of every active true mule in it, as the test keys always score them."""
    active = frame.loc[frame["status"] == ACTIVE, ["t_ms", "account_id", "mule_ring_id"]]
    accounts_of: dict[int, list[str]] = {}
    ring_of: dict[tuple[int, str], int] = {}
    for t_ms, account, ring in active.itertuples(index=False, name=None):
        accounts_of.setdefault(int(t_ms), []).append(str(account))
        ring_of[(int(t_ms), str(account))] = int(ring)
    holds: list[bool] = []
    by_sharing: list[bool] = []
    by_payments: list[bool] = []
    for t_ms, accounts in accounts_of.items():
        rows = read_rows(cache, [(t_ms, account) for account in accounts])
        mules = [
            ActiveMule(t_ms, row.scored, customer_of(row), ring_of[(t_ms, row.scored)])
            for row in rows
        ]
        own = [own_nodes(row) for row in rows]
        found = ring_nodes(mules, own)
        for mule, held, row in zip(mules, own, rows, strict=True):
            holds.append(any((mule.ring, mule.t_ms, node) in found for node in held.nodes))
            by_sharing.append(differs_by_sharing(row.community))
            by_payments.append(differs_by_payments(row.community))
    positives = len(holds)
    with_node = sum(holds)
    endpoints = pd.DataFrame(
        {
            "positives": [positives],
            "with_ring_node": [with_node],
            "share": [with_node / positives if positives else math.nan],
        }
    )
    by_holding = pd.DataFrame(
        {"holds": holds, "sharing": by_sharing, "payments": by_payments}, dtype=bool
    )
    community: list[dict[str, object]] = []
    for held in (True, False):
        chosen = by_holding.loc[by_holding["holds"] == held]
        count = len(chosen)
        community.append(
            {
                "holds_ring_node": held,
                "positives": count,
                "differs_by_sharing": int(chosen["sharing"].sum()),
                "share_by_sharing": float(chosen["sharing"].mean()) if count else math.nan,
                "differs_by_payments": int(chosen["payments"].sum()),
                "share_by_payments": float(chosen["payments"].mean()) if count else math.nan,
            }
        )
    return {ENDPOINTS: endpoints, COMMUNITY: pd.DataFrame(community)}
