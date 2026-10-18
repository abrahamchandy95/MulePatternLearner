"""Reference-world fixtures and the transforms the leak tests apply to a world.

The fixtures are the ordinary look-alikes of the behaviour catalogue beside the mule
behaviours they resemble (docs/design.md, "Leak rules and how tests enforce them", under Other
tests), each a few accounts added to a WorldBuilder around T, whose B1 to B3 rows the tests
work out by hand:

- direct_deposit_split: a salary of which 20% moves to the holder's savings 20 minutes later,
  on rail internal, and the rest leaves over days;
- self_transfer: a saver moving money between its own two accounts;
- dormancy_wake_up: an account quiet for the first 78 days of the lookback, then a refund
  spent over two days;
- landlord: rent from the same three tenants every month;
- family_chain: a parent paying a student, who pays the school within a day;
- new_job: a new employer's first payroll, one new external payer;
- mule_ring: a pass-through mule keeping 7% of its victims' money and forwarding the rest
  within minutes, and the collector it forwards to, which gathers from three mules and sends
  the money on once.

Every customer joined, and every account opened, long before the lookback; HUBS names the
merchants, billers and ledgers a whole world would make account hubs.

The transforms take a world's contract frames and give new ones: erase_labels empties every
label column; move_dates_of_record moves every date of record and link start before the
lookback earlier by a keyed amount; swap_same_second_ids swaps the ids of payments of one
second. A query that reads no label, no date of record as a value and no payment id as an
order answers each world as it answers the original (docs/design.md, the Labels, No dates of
record and Order-free ties guards).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
import hashlib

import pandas as pd

from ..contract import EXTERNAL_LABEL
from .reference.graph import SECOND_MS, ZELLE
from .world import DAY_MS, HOUR_MS, NO_RING, Mule, T, WorldBuilder

MINUTE_MS = 60_000
D, H, M = DAY_MS, HOUR_MS, MINUTE_MS
# When the fixtures' customers joined and their accounts opened, and when the other banks'
# accounts and the ledgers opened: long before the lookback.
JOINED = T - 400 * D
OPENED = T - 900 * D
# The ring of the mule fixtures.
RING = 7
# The merchants, billers and ledgers of the fixtures that a whole world would make account
# hubs: B2 reads them from its parameter, as read_hubs would give them.
HUBS = (
    "x_split_grocer",
    "x_split_utility",
    "x_saver_shop",
    "x_sleeper_power",
    "x_sleeper_mall",
    "g_loans",
    "x_chain_cafe",
    "x_newjob_power",
    "x_newjob_shop",
    "x_mule_shop",
    "x_mule_atm",
    "x_collector_atm",
)


def _externals(w: WorldBuilder, *account_ids: str) -> None:
    for account_id in account_ids:
        w.external(account_id, OPENED)


def direct_deposit_split(w: WorldBuilder) -> str:
    """p_split's checking account a_split: an employer pays 3,000.00 by ACH at 09:00 every two
    weeks, 5 + 14k days before t for k from 0 to 6; 20 minutes after each salary 600.00 moves
    to its own savings, a_split_save, on rail internal; 27 hours after it 1,500.00 goes to a
    grocer by card and 3 days after it 800.00 to a utility by ACH. Its id."""
    w.customer("split", JOINED)
    w.account("a_split_save", "p_split", JOINED)
    _externals(w, "x_split_employer", "x_split_grocer", "x_split_utility")
    for k in range(7):
        paid = T - (5 + 14 * k) * D + 9 * H
        w.pay(f"x_split_s{k}", "x_split_employer", "a_split", paid, 3_000.0, "ach")
        w.pay(f"x_split_m{k}", "a_split", "a_split_save", paid + 20 * M, 600.0, "internal")
        w.pay(f"x_split_c{k}", "a_split", "x_split_grocer", paid + 27 * H, 1_500.0, "card")
        w.pay(f"x_split_b{k}", "a_split", "x_split_utility", paid + 3 * D, 800.0, "ach")
    return "a_split"


def self_transfer(w: WorldBuilder) -> str:
    """p_saver's checking account a_saver: a salary of 3,200.00 by ACH 58 and 28 days before
    t; 250.00 moved to its own savings, a_saver_save, on rail internal 57, 27 and 13 days
    before t; rent of 1,400.00 by ACH 56 and 26 days before t; 100.00 moved back from the
    savings 8 days before t at 21:00, and 120.00 spent by card an hour later. Its id."""
    w.customer("saver", JOINED)
    w.account("a_saver_save", "p_saver", JOINED)
    _externals(w, "x_saver_employer", "x_saver_landlord", "x_saver_shop")
    for n, days in enumerate((58, 28)):
        w.pay(f"x_saver_s{n}", "x_saver_employer", "a_saver", T - days * D + 9 * H, 3_200.0, "ach")
    for n, days in enumerate((57, 27, 13)):
        w.pay(f"x_saver_m{n}", "a_saver", "a_saver_save", T - days * D + 20 * H, 250.0, "internal")
    for n, days in enumerate((56, 26)):
        w.pay(f"x_saver_r{n}", "a_saver", "x_saver_landlord", T - days * D + 10 * H, 1_400.0, "ach")
    w.pay("x_saver_back", "a_saver_save", "a_saver", T - 8 * D + 21 * H, 100.0, "internal")
    w.pay("x_saver_card", "a_saver", "x_saver_shop", T - 8 * D + 22 * H, 120.0, "card")
    return "a_saver"


def dormancy_wake_up(w: WorldBuilder) -> str:
    """p_sleeper's account a_sleeper: a bill of 90.00 by ACH 95 days before t, before the
    lookback, then nothing until a tax refund of 1,500.00 by ACH 12 days before t at 08:00,
    spent by card at a mall: 200.00 at 15:00 that day, 350.00 at 13:00 and 120.00 at 20:00
    the next. Its id."""
    w.customer("sleeper", JOINED)
    _externals(w, "x_sleeper_power", "x_sleeper_treasury", "x_sleeper_mall")
    w.pay("x_sleeper_bill", "a_sleeper", "x_sleeper_power", T - 95 * D + 9 * H, 90.0, "ach")
    w.pay("x_sleeper_refund", "x_sleeper_treasury", "a_sleeper", T - 12 * D + 8 * H, 1_500.0, "ach")
    for n, (at, amount) in enumerate(
        ((T - 12 * D + 15 * H, 200.0), (T - 11 * D + 13 * H, 350.0), (T - 11 * D + 20 * H, 120.0))
    ):
        w.pay(f"x_sleeper_c{n}", "a_sleeper", "x_sleeper_mall", at, amount, "card")
    return "a_sleeper"


def landlord(w: WorldBuilder) -> str:
    """p_landlord's account a_landlord: rent of 1,200.00 from the same three tenants in each of
    the months starting 64, 34 and 4 days before t: a_tenant1 by Zelle that day at 10:00,
    a_tenant2 by Zelle the next day at 19:00, and another bank's x_tenant3 by ACH the day
    after at 08:00; a mortgage of 2,500.00 by ACH to the bank's loan ledger 60 and 30 days
    before t at 09:00; a plumber, a_plumber, paid 300.00 by Zelle the day before t at 15:00.
    Its id."""
    w.customer("landlord", JOINED)
    for name in ("tenant1", "tenant2", "plumber"):
        w.customer(name, JOINED)
    _externals(w, "x_tenant3")
    w.account("g_loans", None, OPENED, account_type="gl")
    for start in (64, 34, 4):
        month = T - start * D
        w.pay(f"z_rent1_{start}", "a_tenant1", "a_landlord", month + 10 * H, 1_200.0, ZELLE)
        w.pay(f"z_rent2_{start}", "a_tenant2", "a_landlord", month + D + 19 * H, 1_200.0, ZELLE)
        w.pay(f"x_rent3_{start}", "x_tenant3", "a_landlord", month + 2 * D + 8 * H, 1_200.0, "ach")
    for days in (60, 30):
        w.pay(f"x_mortgage_{days}", "a_landlord", "g_loans", T - days * D + 9 * H, 2_500.0, "ach")
    w.pay("z_plumber", "a_landlord", "a_plumber", T - D + 15 * H, 300.0, ZELLE)
    return "a_landlord"


def family_chain(w: WorldBuilder) -> str:
    """p_student's account a_student, paid by its parent's a_parent by Zelle at 19:00: 300.00
    50 days before t, 2,000.00 20 days before t and 300.00 6 days before t. The student pays
    the school 1,900.00 by ACH at 11:00 the day after the 2,000.00, and spends at a cafe by
    card: 250.00 45 days before t at 12:00, 60.00 18 days before t at 13:00 and 45.00 5 days
    before t at 12:00. Its id."""
    w.customer("parent", JOINED)
    w.customer("student", JOINED)
    _externals(w, "x_school", "x_chain_cafe")
    for days, amount in ((50, 300.0), (20, 2_000.0), (6, 300.0)):
        at = T - days * D + 19 * H
        w.pay(f"z_allowance_{days}", "a_parent", "a_student", at, amount, ZELLE)
    w.pay("x_tuition", "a_student", "x_school", T - 19 * D + 11 * H, 1_900.0, "ach")
    for days, hour, amount in ((45, 12, 250.0), (18, 13, 60.0), (5, 12, 45.0)):
        at = T - days * D + hour * H
        w.pay(f"x_cafe_{days}", "a_student", "x_chain_cafe", at, amount, "card")
    return "a_student"


def new_job(w: WorldBuilder) -> str:
    """p_newjob's account a_newjob: an old employer's payroll of 2,200.00 by ACH 68, 54 and 40
    days before t at 09:00, then a new employer's first payroll of 2,400.00 by ACH 3 days
    before t at 09:00; rent of 1,100.00 by ACH 62, 32 and 2 days before t at 10:00; a power
    bill of 600.00 by ACH 55 and 25 days before t at 09:00; 80.00 by card at a shop 3 days
    before t at 12:00. p_newjob also holds an old account, a_newjob_old, closed 200 days
    before t. Its id."""
    w.customer("newjob", JOINED)
    w.account("a_newjob_old", "p_newjob", OPENED, closed_ms=T - 200 * D)
    _externals(w, "x_oldco", "x_newco", "x_newjob_landlord", "x_newjob_power", "x_newjob_shop")
    for days in (68, 54, 40):
        w.pay(f"x_oldco_{days}", "x_oldco", "a_newjob", T - days * D + 9 * H, 2_200.0, "ach")
    w.pay("x_newco_3", "x_newco", "a_newjob", T - 3 * D + 9 * H, 2_400.0, "ach")
    for days in (62, 32, 2):
        at = T - days * D + 10 * H
        w.pay(f"x_newjob_rent_{days}", "a_newjob", "x_newjob_landlord", at, 1_100.0, "ach")
    for days in (55, 25):
        at = T - days * D + 9 * H
        w.pay(f"x_newjob_power_{days}", "a_newjob", "x_newjob_power", at, 600.0, "ach")
    w.pay("x_newjob_card", "a_newjob", "x_newjob_shop", T - 3 * D + 12 * H, 80.0, "card")
    return "a_newjob"


def mule_ring(w: WorldBuilder) -> tuple[str, str]:
    """A ring's pass-through mule and its collector. The pass-through mule a_mule, identified,
    keeps its ordinary life: a payroll of 1,800.00 by ACH 75, 45 and 15 days before t at
    09:00, 1,600.00 by card at a shop 73 and 43 days before t at 18:00, 200.00 at an ATM 15
    days before t at 21:00 and 400.00 by card 14 days before t at 18:00. Then its victims pay:
    a_vic1 1,000.00 by Zelle and another bank's x_vic2 500.00 by ACH in the same second, 6
    days before t at 14:00, and it forwards 1,395.00 (93%) to the collector by Zelle 9 minutes
    later; x_vic3 pays 2,000.00 by ACH 4 days before t at 20:00, and it forwards 1,860.00
    (93%) 30 minutes later.

    The collector a_collector, an organizer's account no one identified, also gathers 930.00
    from a_mule2 5 days before t at 10:00 and 465.00 from a_mule3 3 days before t at 16:00, all
    by Zelle; it sends 4,500.00 by ACH to another bank 2 days before t at 12:00 and takes
    100.00 from an ATM the day before t at 02:00. Their ids, the mule's first."""
    victims = (T - 6 * D + 14 * H, T - 4 * D + 20 * H + 30 * M)
    w.customer("mule", JOINED, mule=Mule(*victims, RING, identified=True))
    collected = (T - 6 * D + 14 * H + 9 * M, T - D + 2 * H)
    w.customer("collector", JOINED, mule=Mule(*collected, RING))
    for name, at in (("mule2", T - 5 * D + 10 * H), ("mule3", T - 3 * D + 16 * H)):
        w.customer(name, JOINED, mule=Mule(at, at, RING))
    w.customer("vic1", JOINED)
    _externals(w, "x_vic2", "x_vic3", "x_mule_payroll", "x_mule_shop", "x_mule_atm")
    _externals(w, "x_offshore", "x_collector_atm")
    for days in (75, 45, 15):
        w.pay(f"x_mule_pay{days}", "x_mule_payroll", "a_mule", T - days * D + 9 * H, 1_800.0, "ach")
    for days in (73, 43):
        at = T - days * D + 18 * H
        w.pay(f"x_mule_card{days}", "a_mule", "x_mule_shop", at, 1_600.0, "card")
    w.pay("x_mule_atm15", "a_mule", "x_mule_atm", T - 15 * D + 21 * H, 200.0, "cash")
    w.pay("x_mule_card14", "a_mule", "x_mule_shop", T - 14 * D + 18 * H, 400.0, "card")
    first, second = T - 6 * D + 14 * H, T - 4 * D + 20 * H
    w.pay("z_mule_v1", "a_vic1", "a_mule", first, 1_000.0, ZELLE)
    w.pay("x_mule_v2", "x_vic2", "a_mule", first, 500.0, "ach")
    w.pay("z_mule_f1", "a_mule", "a_collector", first + 9 * M, 1_395.0, ZELLE)
    w.pay("x_mule_v3", "x_vic3", "a_mule", second, 2_000.0, "ach")
    w.pay("z_mule_f2", "a_mule", "a_collector", second + 30 * M, 1_860.0, ZELLE)
    w.pay("z_collector_g2", "a_mule2", "a_collector", T - 5 * D + 10 * H, 930.0, ZELLE)
    w.pay("z_collector_g3", "a_mule3", "a_collector", T - 3 * D + 16 * H, 465.0, ZELLE)
    w.pay("x_collector_sweep", "a_collector", "x_offshore", T - 2 * D + 12 * H, 4_500.0, "ach")
    w.pay("x_collector_cash", "a_collector", "x_collector_atm", T - D + 2 * H, 100.0, "cash")
    return "a_mule", "a_collector"


def catalogue() -> WorldBuilder:
    """Every fixture in one world; none shares an account with another."""
    w = WorldBuilder()
    direct_deposit_split(w)
    self_transfer(w)
    dormancy_wake_up(w)
    landlord(w)
    family_chain(w)
    new_job(w)
    mule_ring(w)
    return w


# The transforms.

LABEL_COLUMNS = ("is_mule", "pu_label", "mule_ring_id", "mule_onset_ms", "mule_last_ms")
# The key of the keyed amounts by which move_dates_of_record moves a date, and the most it
# moves one; 1970-01-02 00:00 UTC, before which no date may fall.
MOVE_KEY = b"mule-move-v1\x00\x00\x00\x00"
MOST_MOVED_MS = 365 * DAY_MS
EARLIEST_MS = DAY_MS
# The dates of record of each vertex table, and the link tables, whose starts move.
DATES_OF_RECORD = {"Party": ("joined_ms",), "Account": ("opened_ms", "closed_ms")}
LINK_TABLES = (
    "Party_Owns_Account",
    "Party_Uses_Token",
    "Token_Bound_To_Account",
    "Party_Uses_Device",
    "Party_Uses_IP",
)
PAYMENT_TABLES = ("Payment_Transaction", "Zelle_Transfer")
# The tables of the roles of each payment table, whose first column is the payment's id.
ROLE_TABLES = {
    "Payment_Transaction": (
        "Transaction_From_Account",
        "Transaction_To_Account",
        "Transaction_Used_Device",
        "Transaction_Used_IP",
    ),
    "Zelle_Transfer": (
        "Transfer_From_Account",
        "Transfer_To_Account",
        "Transfer_From_Token",
        "Transfer_To_Token",
        "Transfer_Used_Device",
        "Transfer_Used_IP",
    ),
}


def erase_labels(tables: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """The world with every label column erased: every internal account reads as no mule
    (is_mule and pu_label 0, no ring, no activity period), and another bank's account keeps
    the convention's -1."""
    accounts = tables["Account"].copy()
    external = accounts["is_external"]
    for column in ("is_mule", "pu_label"):
        accounts[column] = external.map({True: EXTERNAL_LABEL, False: 0}).astype("int64")
    accounts["mule_ring_id"] = NO_RING
    accounts["mule_onset_ms"] = 0
    accounts["mule_last_ms"] = 0
    return {**tables, "Account": accounts}


def _moved(value: int, *key: object) -> int:
    """value moved earlier by an amount of 1 ms to MOST_MOVED_MS keyed by key; never before
    1970-01-02, which no date may precede, and never clamped there."""
    digest = hashlib.blake2b(repr(key).encode(), key=MOVE_KEY, digest_size=8).digest()
    moved = value - 1 - int.from_bytes(digest) % MOST_MOVED_MS
    if moved < EARLIEST_MS:
        raise ValueError(f"Moving {value} by {key} would put it before 1970-01-02")
    return moved


def move_dates_of_record(
    tables: Mapping[str, pd.DataFrame], t_ms: int, lookback_ms: int
) -> dict[str, pd.DataFrame]:
    """The world with every date of record (joined_ms, opened_ms, a closed_ms that is not 0)
    and every link start strictly before t_ms - lookback_ms moved earlier by a keyed amount.
    Each stays on the same side of every bound a query compares it with: t, the lookback's
    start, 30 days before t, and a known start's join, since a start before the lookback is
    never known. Link ends and payment times stay as they are."""
    start = t_ms - lookback_ms
    found = dict(tables)
    for name, columns in DATES_OF_RECORD.items():
        rows = tables[name].copy()
        for column in columns:
            rows[column] = [
                _moved(value, name, row_id, column) if 0 < value < start else value
                for row_id, value in zip(rows["id"], rows[column], strict=True)
            ]
        found[name] = rows
    for name in LINK_TABLES:
        rows = tables[name].copy()
        rows["valid_from_ms"] = [
            _moved(value, name, source, target, value) if value < start else value
            for source, target, value in zip(
                rows["from_id"], rows["to_id"], rows["valid_from_ms"], strict=True
            )
        ]
        found[name] = rows
    return {name: frame.astype(tables[name].dtypes.to_dict()) for name, frame in found.items()}


def _renamed(ids: pd.Series, renamed: Mapping[str, str]) -> pd.Series:
    """A column of payment ids, each renamed, in the column's own dtype."""
    return ids.map(renamed).astype(ids.dtype)


def swap_same_second_ids(
    tables: Mapping[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """The world with the ids of the payments of each second (Payment_Transaction and
    Zelle_Transfer together) given back in reverse order: of the payments of one second,
    sorted by id, the first takes the last's id, the second the one before it, and so on. Each
    payment keeps its amount, rail, time, ends, device and IP. With it, each new id mapped to
    the id its payment had."""
    by_second: dict[int, list[str]] = defaultdict(list)
    for name in PAYMENT_TABLES:
        for payment_id, ts in zip(tables[name]["id"], tables[name]["event_ts_ms"], strict=True):
            by_second[ts // SECOND_MS].append(payment_id)
    renamed: dict[str, str] = {}
    for ids in by_second.values():
        ordered = sorted(ids)
        renamed.update(zip(ordered, reversed(ordered), strict=True))
    found = dict(tables)
    for name in PAYMENT_TABLES:
        found[name] = tables[name].assign(id=_renamed(tables[name]["id"], renamed))
        for role in ROLE_TABLES[name]:
            found[role] = tables[role].assign(from_id=_renamed(tables[role]["from_id"], renamed))
    original = {new: old for old, new in renamed.items()}
    return found, original
