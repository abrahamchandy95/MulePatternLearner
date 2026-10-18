"""read_community, B6: how the account's network formed. For each account of a request at one
Monday, the community of customers around its owner as of t, by sharing in two rounds and then
by payments, and the 8 numbers of its block (docs/design.md, "read_community (B6): how the mule
network formed", and docs/features.md, "The community block")."""

from __future__ import annotations

from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from ...config import DEFAULT_CONFIG
from .rows import account_ids, as_of, parse_row, projected, response_head, vertex_set

# The most members a community holds, the owner included.
COMMUNITY = DEFAULT_CONFIG.subgraph.community


@dataclass(frozen=True)
class Community:
    """A scored account's community block at t (docs/features.md, "The community block").

    - members: the customers kept, its owner included; 0 for an account with no owner at t.
    - capped: a candidate of some round was left out, so the community holds 64.
    - shared_nodes: the devices and IPs that are no leaf and that two members or more share.
      A link is a member sharing one of them; density is the links over members times shared
      nodes, and new_link_share_30d the share of the links whose sharing start is in the last
      30 days. Both are 0 with no shared node.
    - internal_amount_30d: the cents of the payments of the last 30 days between accounts of
      two different members.
    - isolation: the internal payments of the last 30 days over the internal and external
      ones, external meaning with an account of no member; member_payments_30d, their sum.
      isolation is 0 with none.
    """

    account_id: str
    members: int
    capped: bool
    shared_nodes: int
    density: float
    new_link_share_30d: float
    internal_amount_30d: int
    isolation: float
    member_payments_30d: int


SHARES = ("density", "new_link_share_30d", "isolation")


def params(t_ms: int, accounts: Iterable[str], hubs: Iterable[str]) -> dict[str, Any]:
    """The parameters of a request: the Monday t, the accounts, at least one, each once, and
    the account hubs of t, read_hubs' accounts of the same Monday, each once."""
    return {
        "t_ms": as_of(t_ms),
        "account_ids": account_ids(accounts),
        "hub_accounts": vertex_set(hubs),
    }


def _check(row: Community) -> None:
    """Refuse a row whose values the query cannot give."""
    from . import READ_COMMUNITY  # the registry imports this module

    def refuse(why: str) -> ValueError:
        return ValueError(f"{READ_COMMUNITY} answered {row.account_id} with {why}: {row}")

    if row.members > COMMUNITY:
        raise refuse(f"more than {COMMUNITY} members")
    if row.capped and row.members != COMMUNITY:
        raise refuse("a cut community that is not full")
    if not all(0.0 <= getattr(row, name) <= 1.0 for name in SHARES):
        raise refuse("a share outside [0, 1]")
    if row.members == 0 and (
        row.capped or row.shared_nodes or row.internal_amount_30d or row.member_payments_30d
    ):
        raise refuse("a block without an owner that is not empty")
    if row.shared_nodes and row.members < 2:
        raise refuse("a node shared by two members of a community of fewer")
    if (row.density > 0.0) != (row.shared_nodes > 0):
        raise refuse("a density unlike its shared nodes")
    if row.new_link_share_30d and not row.shared_nodes:
        raise refuse("new links without a shared node")
    if row.isolation and not row.member_payments_30d:
        raise refuse("an isolation without a payment")
    if (row.isolation or row.internal_amount_30d) and row.members < 2:
        raise refuse("payments between members of a community of one")
    if row.internal_amount_30d and not row.isolation:
        raise refuse("cents between members without an internal payment")


def parse(rows: Sequence[object], accounts: Collection[str]) -> tuple[Community, ...]:
    """The answer to the request params(t_ms, accounts, hubs) made: a block for every account
    asked for and no other, sorted by id."""
    from . import READ_COMMUNITY, VERSIONS  # the registry imports this module

    response_head(rows, READ_COMMUNITY, VERSIONS[READ_COMMUNITY])
    found = [parse_row(Community, row) for row in projected(rows, "accounts", READ_COMMUNITY)]
    asked = set(accounts)
    answered = [row.account_id for row in found]
    if len(set(answered)) != len(answered) or set(answered) != asked:
        raise ValueError(
            f"{READ_COMMUNITY} answered the accounts {sorted(answered)}, not {sorted(asked)}"
        )
    for row in found:
        _check(row)
    return tuple(sorted(found, key=lambda row: row.account_id))
