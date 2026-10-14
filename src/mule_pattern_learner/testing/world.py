"""A small world for tests, written as PhantomLedger's 27 tables.

WorldBuilder adds parties, accounts, links and payments by hand; standard_world builds the
world the reference graph's tests pin. Times are milliseconds around T.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..tigergraph.queries import WRITE_SPLIT, split_params
from .reference_graph import DAY_MS, LINKS, TABLES, ReferenceGraph

HOUR_MS = 3_600_000
# The as-of time the tests use most: Monday 2024-07-01 00:00 UTC.
T = 1_719_792_000_000
# Four weeks later, a Monday after the late account opened.
LATER = T + 28 * DAY_MS
# Vertices with no attribute but their id, added when a link or payment names them.
_BARE = frozenset({"Token", "Device", "IP", "Address"})


@dataclass(frozen=True)
class Mule:
    """A mule account's answer key and supervision: its ring, place in the chain and flow,
    when it began and last acted, and when and how a bank found it (0 and "" if never)."""

    ring: int
    layer: str
    flow: str
    onset_ms: int
    last_ms: int
    discovered_ms: int = 0
    channel: str = ""


class WorldBuilder:
    """The rows of a world, table by table, and which parties are held out."""

    def __init__(self) -> None:
        self.rows: dict[str, list[tuple[object, ...]]] = {name: [] for name in TABLES}
        self.held_out: dict[str, bool] = {}
        self._bare: set[tuple[str, str]] = set()

    def _vertex(self, table: str, vertex_id: str) -> None:
        if (table, vertex_id) not in self._bare:
            self._bare.add((table, vertex_id))
            self.rows[table].append((vertex_id,))

    def party(self, party_id: str, joined_ms: int, *, held_out: bool = False) -> None:
        """A customer; held out ones get split 1."""
        self.rows["Party"].append((party_id, joined_ms))
        self.held_out[party_id] = held_out

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
    ) -> None:
        """An account, owned by owner from its opening to its closing when it has one. An
        external account's is_mule is -1 (unknown), any other's 1 when mule is given."""
        truth = mule or Mule(-1, "", "", 0, 0)
        is_mule = -1 if external else int(mule is not None)
        self.rows["Account"].append(
            (
                account_id,
                account_type,
                external,
                opened_ms,
                closed_ms,
                is_mule,
                truth.ring,
                truth.layer,
                truth.flow,
                truth.onset_ms,
                truth.last_ms,
                truth.discovered_ms,
                truth.channel,
            )
        )
        if owner is not None:
            self.link("Party_Owns_Account", owner, account_id, opened_ms, closed_ms)

    def link(self, table: str, source: str, target: str, start: int, end: int = 0) -> None:
        """A row of an association table, held over [start, end), end 0 while open."""
        self.rows[table].append((source, target, start, end))
        for vertex_table, vertex_id in zip(LINKS[table], (source, target), strict=True):
            if vertex_table in _BARE:
                self._vertex(vertex_table, vertex_id)

    def device(self, party_id: str, device_id: str, start: int, end: int = 0) -> None:
        self.link("Party_Uses_Device", party_id, device_id, start, end)

    def ip(self, party_id: str, ip_id: str, start: int, end: int = 0) -> None:
        self.link("Party_Uses_IP", party_id, ip_id, start, end)

    def address(self, party_id: str, address_id: str, start: int, end: int = 0) -> None:
        self.link("Party_Has_Address", party_id, address_id, start, end)

    def token(self, token_id: str, party_id: str, account_id: str, start: int) -> None:
        """A Zelle token the party registered for the account."""
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
    ) -> None:
        """A payment: a Zelle_Transfer when rail is "zelle", else a Payment_Transaction."""
        if rail == "zelle":
            prefix = "Transfer"
            self.rows["Zelle_Transfer"].append((payment_id, amount, ts))
        else:
            prefix = "Transaction"
            self.rows["Payment_Transaction"].append((payment_id, amount, rail, ts))
        self.rows[f"{prefix}_From_Account"].append((payment_id, sender, ts))
        self.rows[f"{prefix}_To_Account"].append((payment_id, receiver, ts))
        for role, table, node in (("Used_Device", "Device", device), ("Used_IP", "IP", ip)):
            if node is not None:
                self._vertex(table, node)
                self.rows[f"{prefix}_{role}"].append((payment_id, node, ts))

    def tables(self) -> dict[str, pd.DataFrame]:
        return {
            name: pd.DataFrame(rows, columns=list(TABLES[name])) for name, rows in self.rows.items()
        }

    def splits(self) -> dict[str, int]:
        """Each party's split: 1 when held out, else 0."""
        return {party_id: int(held) for party_id, held in self.held_out.items()}

    def write_csv(self, directory: Path) -> None:
        """One <table>.csv per table with its header line, booleans written True and False
        as PhantomLedger writes them."""
        for name, frame in self.tables().items():
            frame.to_csv(directory / f"{name}.csv", index=False)

    def graph(self) -> ReferenceGraph:
        """The reference graph of the world, with every party's split written."""
        graph = ReferenceGraph(self.tables())
        if self.held_out:
            [answer] = graph.run(WRITE_SPLIT, split_params(self.splits()))
            assert answer["status"] == "ok", answer
        return graph


def standard_world() -> WorldBuilder:
    """The world the reference graph's tests pin, around T (Monday 2024-07-01).

    - Ann and Ben share a home: its address H_HOME and its IP I_HOME. Ann has a deposit
      account with a Zelle token and a credit card; Ben enrolled the device D_BEN at T.
      Ann paid Ben at the window's start (T01), a second before it (T02), and twice since;
      Ben's payment back (T05) is at T itself.
    - Old's account closed 20 days before T, after it paid Ann.
    - Ring 0: the victim Vic paid Mia (Zelle), who passed it on to Max within two hours, who
      cashed it out to the external X_CASH. Mia and Max hold the ring's device D_RING0
      (Mia's link ended 9 days before T); the ring's IP I_RING0 is linked to no one.
    - Ring 1: Roy paid Sue; both hold D_RING1, Roy until 5 days after T.
    - Hal is held out: he holds D_RING0 and paid Mia on it.
    - Quy's account has no payments and Quy holds D_RING0.
    - New joins 14 days after T, holds D_RING1 and pays Roy 20 days after T.
    - G_FEES is the bank's fee ledger (gl).
    """
    w = WorldBuilder()
    day = DAY_MS
    w.party("P_ANN", T - 400 * day)
    w.account("A_ANN", "P_ANN", T - 400 * day)
    w.account("A_ANN_CARD", "P_ANN", T - 300 * day, account_type="credit")
    w.token("K_ANN", "P_ANN", "A_ANN", T - 200 * day)
    w.device("P_ANN", "D_ANN", T - 400 * day)
    w.ip("P_ANN", "I_HOME", T - 400 * day)
    w.address("P_ANN", "H_HOME", T - 400 * day)
    w.party("P_BEN", T - 350 * day)
    w.account("A_BEN", "P_BEN", T - 350 * day)
    w.device("P_BEN", "D_BEN", T)
    w.ip("P_BEN", "I_HOME", T - 350 * day)
    w.address("P_BEN", "H_HOME", T - 350 * day)
    w.party("P_OLD", T - 500 * day)
    w.account("A_OLD", "P_OLD", T - 500 * day, closed_ms=T - 20 * day)
    w.party("P_VIC", T - 365 * day)
    w.account("A_VIC", "P_VIC", T - 365 * day)
    w.device("P_VIC", "D_VIC", T - 365 * day)

    w.party("P_MIA", T - 300 * day)
    w.account(
        "A_MIA",
        "P_MIA",
        T - 300 * day,
        mule=Mule(
            0,
            "first_layer",
            "pass_through",
            T - 60 * day,
            T - 10 * day + 2 * HOUR_MS,
            T - 5 * day,
            "victim_report",
        ),
    )
    w.device("P_MIA", "D_RING0", T - 62 * day, T - 9 * day)
    w.party("P_MAX", T - 250 * day)
    w.account(
        "A_MAX",
        "P_MAX",
        T - 250 * day,
        mule=Mule(0, "onward_layer", "cash_out", T - 55 * day, T - 8 * day, T + 20 * day, "trace"),
    )
    w.device("P_MAX", "D_RING0", T - 58 * day)
    w.party("P_ROY", T - 200 * day)
    w.account(
        "A_ROY",
        "P_ROY",
        T - 200 * day,
        mule=Mule(
            1, "first_layer", "collector", T - 45 * day, T + 20 * day, T + 30 * day, "monitoring"
        ),
    )
    w.device("P_ROY", "D_RING1", T - 45 * day, T + 5 * day)
    w.party("P_SUE", T - 150 * day)
    w.account(
        "A_SUE",
        "P_SUE",
        T - 150 * day,
        mule=Mule(1, "onward_layer", "received_only", T - 30 * day, T - 30 * day),
    )
    w.device("P_SUE", "D_RING1", T - 40 * day)

    w.party("P_HAL", T - 200 * day, held_out=True)
    w.account("A_HAL", "P_HAL", T - 200 * day)
    w.device("P_HAL", "D_RING0", T - 50 * day)
    w.party("P_QUY", T - 90 * day)
    w.account("A_QUY", "P_QUY", T - 90 * day)
    w.device("P_QUY", "D_RING0", T - 45 * day)
    w.party("P_NEW", T + 14 * day)
    w.account("A_NEW", "P_NEW", T + 14 * day)
    w.device("P_NEW", "D_RING1", T + 14 * day)
    w.account("X_CASH", None, T - 8 * day, account_type="unknown", external=True)
    w.account("G_FEES", None, T - 15 * day, account_type="gl")

    for payment_id, ts, amount, rail in (
        ("T01", T - 365 * day, 100.0, "ach"),
        ("T02", T - 365 * day - 1000, 75.0, "ach"),
        ("T03", T - 40 * day, 120.0, "ach"),
        ("T04", T - 6 * day, 80.0, "zelle"),
    ):
        w.pay(payment_id, "A_ANN", "A_BEN", ts, amount, rail, device="D_ANN", ip="I_HOME")
    w.pay("T05", "A_BEN", "A_ANN", T, 60.0, "ach")
    w.pay("T06", "A_OLD", "A_ANN", T - 25 * day, 50.0, "check")
    w.pay("T07", "A_ANN_CARD", "G_FEES", T - 15 * day, 25.0, "internal")
    w.pay("T08", "A_VIC", "A_MIA", T - 10 * day, 900.0, "zelle", device="D_VIC", ip="I_VIC")
    nine = T - 10 * day + 2 * HOUR_MS
    w.pay("T09", "A_MIA", "A_MAX", nine, 870.0, "zelle", device="D_RING0", ip="I_RING0")
    w.pay("T10", "A_MAX", "X_CASH", T - 8 * day, 850.0, "cash", device="D_RING0")
    w.pay("T11", "A_HAL", "A_MIA", T - 20 * day, 300.0, "ach", device="D_RING0")
    w.pay("T12", "A_ROY", "A_SUE", T - 30 * day, 400.0, "zelle", device="D_RING1")
    w.pay("T13", "A_NEW", "A_ROY", T + 20 * day, 200.0, "zelle", device="D_RING1")
    return w
