"""Worlds for tests, written as the 22 tables of the export contract (contract.py), as
PhantomLedger's mule use case exports them.

WorldBuilder adds customers, accounts, devices, IPs, tokens, links and payments by hand, with
helpers for the shapes tests build again and again: a household, a ring sharing a device, a
joiner whose first links start at its first use, and another bank's account. Times are UTC
milliseconds; T, the time tests use most, is 2024-07-01 00:00 UTC, the Monday that starts a scored
week.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass

import pandas as pd

from ..contract import EXTERNAL_LABEL, TABLES
from .reference.graph import (
    BY_NAME,
    DEVICE,
    IP,
    ZELLE,
    ReferenceGraph,
    frame,
)

HOUR_MS = 3_600_000
DAY_MS = 24 * HOUR_MS
# Monday 2024-07-01 00:00 UTC, the start of the 14th of the 52 scored weeks.
T = 1_719_792_000_000
# The ring id of a mule with no ring, and of every account that is no mule.
NO_RING = -1
# The vertex types a link or payment adds when it names one, since they have no attribute but
# their id.
_BARE = frozenset({"Token", DEVICE, IP})


@dataclass(frozen=True)
class Mule:
    """A mule account's truth: its activity period (its first and last ring payment, or a
    recruit's recruitment and its leaving the ring), its ring (NO_RING for none), and whether
    the bank identified it (pu_label 1)."""

    onset_ms: int
    last_ms: int
    ring: int = NO_RING
    identified: bool = False


class WorldBuilder:
    """The rows of a world, table by table."""

    def __init__(self) -> None:
        self.rows: dict[str, list[tuple[object, ...]]] = {table.name: [] for table in TABLES}
        self._bare: set[tuple[str, str]] = set()

    def _vertex(self, kind: str, vertex_id: str) -> None:
        if (kind, vertex_id) not in self._bare:
            self._bare.add((kind, vertex_id))
            self.rows[kind].append((vertex_id,))

    def party(self, party_id: str, joined_ms: int) -> str:
        """A customer, who joined the bank at joined_ms."""
        self.rows["Party"].append((party_id, joined_ms))
        return party_id

    def account(
        self,
        account_id: str,
        owner: str | None,
        opened_ms: int,
        *,
        account_type: str = "deposit",
        closed_ms: int = 0,
        external: bool = False,
        mule: Mule | None = None,
    ) -> str:
        """An account, owned by owner from its opening to its closing (0 while open), as the
        loader requires of an owned account. Labels follow the convention: is_mule and pu_label
        -1 on another bank's account, which has no owner and no truth; otherwise is_mule 1 for
        a mule and pu_label 1 for an identified one, else 0, with a ring and activity period
        on a mule only."""
        if external and (owner is not None or mule is not None):
            raise ValueError(f"{account_id} is another bank's account, with no owner or truth")
        if external:
            labels = (EXTERNAL_LABEL, EXTERNAL_LABEL, NO_RING, 0, 0)
        elif mule is None:
            labels = (0, 0, NO_RING, 0, 0)
        else:
            labels = (1, int(mule.identified), mule.ring, mule.onset_ms, mule.last_ms)
        row = (account_id, account_type, external, opened_ms, closed_ms, *labels)
        self.rows["Account"].append(row)
        if owner is not None:
            self.link("Party_Owns_Account", owner, account_id, opened_ms, closed_ms)
        return account_id

    def link(self, table: str, source: str, target: str, start: int, end: int = 0) -> None:
        """A row of a link table, held from start until end, 0 while open."""
        self.rows[table].append((source, target, start, end))
        found = BY_NAME[table]
        for kind, vertex_id in ((found.source, source), (found.target, target)):
            if kind in _BARE:
                self._vertex(kind, vertex_id)

    def device(self, party_id: str, device_id: str, start: int, end: int = 0) -> None:
        """The customer's enrolment on a device."""
        self.link("Party_Uses_Device", party_id, device_id, start, end)

    def ip(self, party_id: str, ip_id: str, start: int, end: int = 0) -> None:
        """The customer's enrolment on an IP."""
        self.link("Party_Uses_IP", party_id, ip_id, start, end)

    def token(self, token_id: str, party_id: str, account_id: str, start: int) -> None:
        """A Zelle token the customer enrolled and bound to the account at start."""
        self.link("Party_Uses_Token", party_id, token_id, start)
        self.link("Token_Bound_To_Account", token_id, account_id, start)

    def pay(
        self,
        payment_id: str,
        sender: str,
        receiver: str,
        ts: int,
        amount: float,
        rail: str,
        *,
        device: str | None = None,
        ip: str | None = None,
        tokens: tuple[str, str] | None = None,
    ) -> None:
        """A payment: a Zelle_Transfer when rail is "zelle", between the tokens (sender's,
        receiver's) when given, else a Payment_Transaction of that payment_rail; seen on the
        device and the IP when given."""
        if rail == ZELLE:
            prefix = "Transfer"
            self.rows["Zelle_Transfer"].append((payment_id, amount, ts))
            for role, token in zip(("From", "To"), tokens, strict=True) if tokens else ():
                self._vertex("Token", token)
                self.rows[f"Transfer_{role}_Token"].append((payment_id, token, ts))
        elif tokens is not None:
            raise ValueError(f"{payment_id} is no Zelle transfer, so it has no tokens")
        else:
            prefix = "Transaction"
            self.rows["Payment_Transaction"].append((payment_id, amount, rail, ts))
        self.rows[f"{prefix}_From_Account"].append((payment_id, sender, ts))
        self.rows[f"{prefix}_To_Account"].append((payment_id, receiver, ts))
        for kind, node in ((DEVICE, device), (IP, ip)):
            if node is not None:
                self._vertex(kind, node)
                self.rows[f"{prefix}_Used_{kind}"].append((payment_id, node, ts))

    # Helpers for the shapes tests build again and again.

    def customer(
        self,
        name: str,
        joined_ms: int,
        *,
        account_type: str = "deposit",
        closed_ms: int = 0,
        mule: Mule | None = None,
    ) -> str:
        """The customer p_<name>, who joined at joined_ms, with the account a_<name> opened
        then. The account's id."""
        party = self.party(f"p_{name}", joined_ms)
        return self.account(
            f"a_{name}",
            party,
            joined_ms,
            account_type=account_type,
            closed_ms=closed_ms,
            mule=mule,
        )

    def household(self, names: Sequence[str], joined_ms: int, device: str, ip: str) -> list[str]:
        """The customers of one home, each with a deposit account, who joined together and
        have been enrolled on the home's device and IP since joining. Their accounts' ids."""
        accounts: list[str] = []
        for name in names:
            accounts.append(self.customer(name, joined_ms))
            self.device(f"p_{name}", device, joined_ms)
            self.ip(f"p_{name}", ip, joined_ms)
        return accounts

    def ring(
        self,
        ring: int,
        names: Sequence[str],
        joined_ms: int,
        device: str,
        start_ms: int,
        onset_ms: int,
        last_ms: int,
        *,
        identified: Collection[str] = (),
    ) -> list[str]:
        """The mule accounts of one ring: customers who joined at joined_ms, each with a
        deposit account that is a mule of the ring active from onset_ms to last_ms, identified
        when its name is in identified, and each enrolled on the ring's device from start_ms,
        since a ring device belongs to the controller who operates the accounts. Their
        accounts' ids."""
        accounts: list[str] = []
        for name in names:
            mule = Mule(onset_ms, last_ms, ring, name in identified)
            accounts.append(self.customer(name, joined_ms, mule=mule))
            self.device(f"p_{name}", device, start_ms)
        return accounts

    def joiner(self, name: str, joined_ms: int, device: str, ip: str, first_use_ms: int) -> str:
        """A customer who joins at joined_ms and first uses a device and an IP at
        first_use_ms, hours or days later: PhantomLedger builds links from observed uses, so
        its first links start at that use, not at joining. Its account's id."""
        account = self.customer(name, joined_ms)
        self.device(f"p_{name}", device, first_use_ms)
        self.ip(f"p_{name}", ip, first_use_ms)
        return account

    def external(self, account_id: str, opened_ms: int, *, account_type: str = "unknown") -> str:
        """Another bank's account, which the bank sees only on payments."""
        return self.account(account_id, None, opened_ms, account_type=account_type, external=True)

    def tables(self) -> dict[str, pd.DataFrame]:
        """The contract frames of the world, one per table."""
        return {name: frame(name, rows) for name, rows in self.rows.items()}

    def graph(self) -> ReferenceGraph:
        """The reference graph of the world."""
        return ReferenceGraph(self.tables())
