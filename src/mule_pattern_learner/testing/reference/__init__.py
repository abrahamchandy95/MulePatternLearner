"""The reference graph: the test oracle that answers the installed queries as TigerGraph does,
written from docs/design.md and docs/features.md apart from the GSQL (docs/design.md, "Package
layout"). graph.py holds its core, and each other module the answers of some queries, which
ANSWERS names by query; ReferenceGraph.run answers through it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ...tigergraph.queries import (
    LIST_ACCOUNTS,
    PROFILE_HUBS,
    READ_ACTIVITY_CHANGE,
    READ_ANSWER_KEY,
    READ_COUNTERPARTIES,
    READ_HUBS,
    READ_LABELS,
    READ_MONEY_FLOW,
    READ_WORLD,
)
from . import activity, counterparties, infrastructure, money_flow
from .graph import ReferenceGraph

# Each query's answer: given the graph and the query's parameters by name, the rows it prints.
ANSWERS: dict[str, Callable[..., list[dict[str, Any]]]] = {
    LIST_ACCOUNTS: infrastructure.list_accounts,
    READ_LABELS: infrastructure.read_labels,
    READ_ANSWER_KEY: infrastructure.read_answer_key,
    READ_HUBS: infrastructure.read_hubs,
    READ_WORLD: infrastructure.read_world,
    PROFILE_HUBS: infrastructure.profile_hubs,
    READ_MONEY_FLOW: money_flow.read_money_flow,
    READ_COUNTERPARTIES: counterparties.read_counterparties,
    READ_ACTIVITY_CHANGE: activity.read_activity_change,
}

__all__ = ["ANSWERS", "ReferenceGraph"]
