"""PhantomLedger's records of a world, as mule audit checks them before a result may be reported
(docs/design.md, "Gates" and "What PhantomLedger must provide").

For every world PhantomLedger writes a world record and a gate record, and the loader writes
them, after a passing verification, into the records folder .env names (MULE_PL_RECORDS_DIR):
world_<seed>.identity.toml, the world record's keys, which data.worlds matches to the loaded
world, and world_<seed>.gates.toml, the gate record. A reported result needs, for each world:

- the identity file's fingerprint equal to read_world's on the fields the two share
  (tigergraph.world.shared_fields), every one of them present;
- the identity file's clean_tree and code_gates_certified true: PhantomLedger built from a
  clean tree, with every code gate of its commit certified;
- in the gate record, every code gate passed, and every gate GATES maps to a feature present
  and passed. A world check of a gate that reaches no feature (labels or evaluation only, or
  nothing) may fail without stopping a reported result, as docs/features.md's gate tables say.

The gate record holds one row per gate, pass or fail only, never a value or a bound:

    [[gate]]
    entry = "S-TIM-1"
    kind = "world"
    pass = true

A gate that runs both as a code gate and as a world check has one row of each kind. Its pass
flags are computed from the truth, so only evaluation/ and testing/ open it (paths.pl_gate_record).

GATES is docs/features.md's gate tables, copied row by row: the dossier entries of "Generator
gates each feature depends on" by their entry ids, the residual checks of "Shortcuts noted but
not yet researched", and PhantomLedger's world checks of "Gates the features add". The design
gives the residual checks and the added world checks no entry id, so this module names them
(R-... and G-...); PhantomLedger's gate record must use the same names, and until it does those
gates read as missing and no result is reported. tests/evaluation/test_gate_records.py holds
GATES to docs/features.md's tables, cell for cell.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import tomllib

from ..data.worlds import read_record, shared_record
from ..tigergraph.queries.read_world import World
from ..tigergraph.world import shared_fields

# The kinds of gate a gate record holds.
CODE = "code"
WORLD = "world"
KINDS = (CODE, WORLD)
# The identity file's keys that say how PhantomLedger was built.
CLEAN_TREE = "clean_tree"
CERTIFIED = "code_gates_certified"


@dataclass(frozen=True)
class Gate:
    """A gate of docs/features.md's gate tables: the entry the gate record names it by, the
    first cell of its row there, the features its row says it reaches, and whether that is a
    feature at all (a gate reaching only labels, evaluation or nothing is not)."""

    entry: str
    row: str
    reaches: str
    feature: bool = True


# docs/features.md's gate tables, row by row.
GATES: tuple[Gate, ...] = (
    Gate(
        "S-TIM-1",
        "S-TIM-1",
        ("every duration measured to t; `dwell`, `median_dwell`, gaps, `left_15m` (whole minutes)"),
    ),
    Gate(
        "S-TIM-2",
        "S-TIM-2",
        (
            "`dwell`, `median_dwell`, `left_15m` to `left_1d`, `sender_gap` of exactly 86,400 s, "
            "bursts, B1 lots"
        ),
    ),
    Gate(
        "S-TIM-3",
        "S-TIM-3",
        "`age` and `receiver_gap` of victim credits, `burst_in_*`, `max_new_payers_24h`",
    ),
    Gate(
        "S-TIM-4",
        "S-TIM-4",
        "`age`, `since_last_out`, `dwell`, `left_*`",
    ),
    Gate(
        "S-TIM-5",
        "S-TIM-5",
        "`age`, `pair_gap`, `sender_gap`, `receiver_gap`",
    ),
    Gate(
        "S-TIM-6",
        "S-TIM-6",
        "`age` and `since_last_out` with rail cash, `left_by_cash`",
    ),
    Gate(
        "S-TIM-7",
        "S-TIM-7",
        "`age`, `pair_age`, `since_first_use`, `since_start`, community timing",
    ),
    Gate(
        "S-TIM-8",
        "S-TIM-8",
        "pair fields between members, `new_payer_share_30d`, `internal_amount_30d`, `isolation`",
    ),
    Gate(
        "S-TIM-9",
        "S-TIM-9",
        "`age`, `receiver_gap` of pushes into unlabelled payees",
    ),
    Gate(
        "S-TIM-10",
        "S-TIM-10",
        "`age`, `since_last_out` of outflows",
    ),
    Gate(
        "S-AMT-1",
        "S-AMT-1",
        "`amount`, `in_amount_*`",
    ),
    Gate(
        "S-AMT-2",
        "S-AMT-2",
        "`amount`, `in_amount_*`, piece sizes",
    ),
    Gate(
        "S-AMT-3",
        "S-AMT-3",
        "`amount` on outflows, `out_amount_*`, out counts, `burst_out_*`",
    ),
    Gate(
        "S-AMT-4",
        "S-AMT-4",
        "`amount` with rail card (kept on edges to flag-only merchants), `left_by_card`",
    ),
    Gate(
        "S-AMT-5",
        "S-AMT-5",
        "`amount`",
    ),
    Gate(
        "S-AMT-6",
        "S-AMT-6",
        "`amount` on cash; no gate",
    ),
    Gate(
        "S-AMT-7",
        "S-AMT-7",
        "`amount` with rail cash, `in_amount_*`",
    ),
    Gate(
        "S-AMT-8",
        "S-AMT-8",
        "`held_share`, `left_7d`, `left_24h`, `dwell`, `median_dwell`",
    ),
    Gate(
        "S-AMT-9",
        "S-AMT-9",
        "`amount`, `pair_prior_count`, `new_payer_share_30d`",
    ),
    Gate(
        "S-AMT-10",
        "S-AMT-10",
        "`amount`",
    ),
    Gate(
        "S-AMT-11",
        "S-AMT-11",
        "`amount`, `in_amount_*`, `out_amount_*`",
    ),
    Gate(
        "S-RAIL-1",
        "S-RAIL-1",
        "`rail`, `left_by_*`",
    ),
    Gate(
        "S-RAIL-2",
        "S-RAIL-2",
        "`rail`, ownership edges, `dwell`, `left_24h`, `left_15m`, `left_by_transfer`",
    ),
    Gate(
        "S-RAIL-3",
        "S-RAIL-3",
        "`rail` cash between two customer accounts, `amount`",
    ),
    Gate(
        "S-RAIL-4",
        "S-RAIL-4",
        "none, since no query reads a self-payment; the loader counts them",
        feature=False,
    ),
    Gate(
        "S-RAIL-5",
        "S-RAIL-5",
        "`amount` with rail cash, `sender_gap` of exactly a day, `out_amount_*`",
    ),
    Gate(
        "S-RAIL-6",
        "S-RAIL-6",
        "`age` of cash and internal edges, once a monthly payment fixes the day of month",
    ),
    Gate(
        "S-RAIL-7",
        "S-RAIL-7",
        "`rail` on other accounts of a customer, their B1 to B3",
    ),
    Gate(
        "S-RAIL-8",
        "S-RAIL-8",
        "seen edges, `accounts_seen_30d`, `payments_seen_30d`",
    ),
    Gate(
        "S-RAIL-9",
        "S-RAIL-9",
        "`rail` of credits",
    ),
    Gate(
        "S-RAIL-10",
        "S-RAIL-10",
        "`is_external` of payees, `new_payees_*`, `top_payee_share_30d`",
    ),
    Gate(
        "S-ID-1",
        "S-ID-1",
        "`co_users`, `co_users_both`, `new_co_users_30d`, community block",
    ),
    Gate(
        "S-ID-2",
        "S-ID-2",
        "`co_users` and community of organizers and collectors",
    ),
    Gate(
        "S-ID-3",
        "S-ID-3",
        (
            "`new_devices_30d`, `new_ips_30d`, `since_start`, `new_customers_*`, "
            "`new_link_share_30d`, early warning, and the No link ends guard"
        ),
    ),
    Gate(
        "S-ID-4",
        "S-ID-4",
        "seen edges, `observed_only`, `co_users`",
    ),
    Gate(
        "S-ID-5",
        "S-ID-5",
        "`uses_90d`, `accounts_seen_30d`, `payments_seen_30d`",
    ),
    Gate(
        "S-ID-6",
        "S-ID-6",
        "`co_users`, `co_users_both`, community block",
    ),
    Gate(
        "S-ID-7",
        "S-ID-7",
        "IP `since_start`, `customers_90d`, `new_customers_*`, `since_first_use`",
    ),
    Gate(
        "S-ID-8",
        "S-ID-8",
        "`since_start`, `new_devices_30d`, `new_ips_30d`, `new_customers_*`",
    ),
    Gate(
        "S-ID-9",
        "S-ID-9",
        "`observed_only`, `crowd`, `customers_90d`",
    ),
    Gate(
        "S-MEM-1",
        "S-MEM-1",
        (
            "`age`, `payees_30d`, `new_payees_*`, `hub_payee_share_30d`, `new_external_payers_*` "
            "(a second payroll)"
        ),
    ),
    Gate(
        "S-MEM-2",
        "S-MEM-2",
        "labels; community `members`",
    ),
    Gate(
        "S-MEM-3",
        "S-MEM-3",
        "`top_payee_share_30d`, collectors' `payers_30d` and hub status",
    ),
    Gate(
        "S-MEM-4",
        "S-MEM-4",
        "labels",
        feature=False,
    ),
    Gate(
        "S-MEM-5",
        "S-MEM-5",
        "labels; their B1 to B3 as unlabelled look-alikes",
    ),
    Gate(
        "S-MEM-6",
        "S-MEM-6",
        "`co_users`, community `density`, `isolation`, `internal_amount_30d`",
    ),
    Gate(
        "S-MEM-7",
        "S-MEM-7",
        "new internal and external payers, payers' rows",
    ),
    Gate(
        "S-MEM-8",
        "S-MEM-8",
        "in counts, `flow_present`, `held_share`",
    ),
    Gate(
        "S-MEM-9",
        "S-MEM-9",
        "`top_payee_share_30d`",
    ),
    Gate(
        "S-LIFE-1",
        "S-LIFE-1",
        "Known starts, `mule prepare`'s lookback check",
    ),
    Gate(
        "S-LIFE-2",
        "S-LIFE-2",
        (
            "positives per Monday, the label rate: the burn-in, the representative start and the "
            "stationarity gates keep the weekly active-ring count flat"
        ),
    ),
    Gate(
        "S-LIFE-3",
        "S-LIFE-3",
        "`age`, `pair_age`, positives per Monday",
    ),
    Gate(
        "S-LIFE-4",
        "S-LIFE-4",
        (
            "`accounts_owned`, newness from counts (`in_90d` against `in_30d`), new payers, "
            "`pair_age`, `since_first_use`"
        ),
    ),
    Gate(
        "S-LIFE-5",
        "S-LIFE-5",
        "`accounts_owned`, the population, the newness lift",
    ),
    Gate(
        "S-LIFE-6",
        "S-LIFE-6",
        "labels, joiner mules' counts",
    ),
    Gate(
        "S-LIFE-7",
        "S-LIFE-7",
        "`held_share` 1 on received-only mules, `flow_present`, `left_*`, empty tenures",
    ),
    Gate(
        "S-LIFE-8",
        "S-LIFE-8",
        (
            "`since_last_in`, `since_last_out` jumping together across members, "
            "`internal_amount_30d`"
        ),
    ),
    Gate(
        "S-LIFE-9",
        "S-LIFE-9",
        "`new_link_share_30d`, `since_start` of ring links, positives per Monday",
    ),
    Gate(
        "S-IDN-1",
        "S-IDN-1 to S-IDN-5",
        "labels and evaluation only; S-IDN-2 settled (identification closes nothing)",
        feature=False,
    ),
    Gate(
        "S-IDN-2",
        "S-IDN-1 to S-IDN-5",
        "labels and evaluation only; S-IDN-2 settled (identification closes nothing)",
        feature=False,
    ),
    Gate(
        "S-IDN-3",
        "S-IDN-1 to S-IDN-5",
        "labels and evaluation only; S-IDN-2 settled (identification closes nothing)",
        feature=False,
    ),
    Gate(
        "S-IDN-4",
        "S-IDN-1 to S-IDN-5",
        "labels and evaluation only; S-IDN-2 settled (identification closes nothing)",
        feature=False,
    ),
    Gate(
        "S-IDN-5",
        "S-IDN-1 to S-IDN-5",
        "labels and evaluation only; S-IDN-2 settled (identification closes nothing)",
        feature=False,
    ),
    Gate(
        "R-BURSTS",
        (
            "the burst model ran only in pure-mule rings (answered by the activity research: one "
            "active period per account, shared by every playbook)"
        ),
        "counts, bursts, `since_last_*`",
    ),
    Gate(
        "R-REPOSTS",
        "a declined ring row may post again at a different second",
        "`dwell`, `age`",
    ),
    Gate(
        "R-MOVE-SIZES",
        (
            "mule own-account moves sized like onward pieces (about USD 662) against LN(USD 120, "
            "0.75) for ordinary self-transfers"
        ),
        "`amount` on internal same-owner edges",
    ),
    Gate(
        "R-RAIL-2",
        "the S-RAIL-2 residual: an internal same-owner move minutes after a non-salary credit",
        "`rail`, ownership edges, `dwell`, `left_24h`, `left_15m`, `left_by_transfer`",
    ),
    Gate(
        "R-MULTI-RING",
        "multi-ring reuse drawn uniformly (p 0.06)",
        "community block",
    ),
    Gate(
        "R-SHELLS",
        (
            "shell accounts as organizers' personal primaries (the shell routing of layering and "
            "scatter-gather stays as behaviour)"
        ),
        "organizers' rows: their B1 to B3, `co_users`, community block",
    ),
    Gate(
        "R-HOURS",
        "legitimate hour profiles uncited",
        "every duration measured to t",
    ),
    Gate(
        "R-PAYDAYS",
        (
            "the payday skew of legitimate ATM days (kept as behaviour, tied to each person's own "
            "paydays)"
        ),
        "`age` of cash edges",
    ),
    Gate(
        "R-CASH-CAP",
        "the consumer cash-deposit cap",
        "`amount` with rail cash",
    ),
    Gate(
        "R-COVER-VOLUME",
        "cover traffic volume (cover kept as behaviour, rebuilt on the holder's own life)",
        "out counts, `payees_30d`, `new_payees_*`",
    ),
    Gate(
        "R-CONTROLLER-COVER",
        (
            "controller-made cover on accounts the holder no longer operates (P2P to a P2P "
            "platform, bills and purchases, on the controller's clock and devices)"
        ),
        (
            "out counts, `payees_30d`, `new_payees_*`; on the ring device, `uses_90d`, "
            "`accounts_seen_30d`, `payments_seen_30d`"
        ),
    ),
    Gate(
        "R-SECOND-JOBS",
        (
            "no legitimate second jobs (PhantomLedger adds them for every person at the BLS rate; "
            "whether ring members keep the cover's 12% is open with the owner)"
        ),
        "`new_external_payers_*`",
    ),
    Gate(
        "R-ATO-CLOCK",
        "the attacker's own clock on ATO drains (kept as behaviour)",
        "`age`",
    ),
    Gate(
        "G-TENURES",
        (
            "every `Party_Uses_Device` and `Party_Uses_IP` tenure built from uses, ring or "
            "ordinary, has no quiet stretch longer than 21 days (a ring tenure's lead counts as "
            "quiet) and ends at most four days after the start of its last use's day; tenures are "
            "derived from every simulated use, burn-in included"
        ),
        "keeps the visible and current tests decided by the past for every party link",
    ),
    Gate(
        "G-OPENINGS",
        "no exported payment comes before its accounts open",
        "every account node exists at t",
    ),
    Gate(
        "G-PAYMENT-IDS",
        (
            "payment ids are pseudonyms whose order does not follow event time; among payments "
            "sharing a second, the chance that a laundering row's id sorts before a legitimate "
            "row's lies within its binomial band of 0.5"
        ),
        "subgraph cap ties carry no class",
    ),
    Gate(
        "G-EXTERNAL-PAIRS",
        "no payment between two external accounts",
        "a bank cannot see them",
    ),
    Gate(
        "G-1970",
        (
            "no exported time before 1970-01-02 00:00 UTC: backdated relationship, account and "
            "link starts are drawn from their distributions truncated there by redrawing, never by"
            " clamping"
        ),
        "the times are UINT milliseconds, 0 for none",
    ),
    Gate(
        "G-NEWNESS",
        (
            'the lift of "no payment in the first 60 days of the lookback, some after", and the '
            "share of customers new to the bank within a year"
        ),
        "account newness rebuilt from behaviour must be as strong as in reality, not more",
    ),
    Gate(
        "G-SAME-OWNER",
        (
            'the lift of "an internal move to an account of the same owner within 1 hour of a '
            'non-payroll credit", and the distance between mule and ordinary same-owner move '
            "amounts"
        ),
        "the same-owner move residual and own-account move sizes",
    ),
    Gate(
        "G-SHARED-LINKS",
        (
            'the lift of "a link to a node shared with another customer, started in the last 14 '
            'days"'
        ),
        "every ring tenure is followed by ring money in PhantomLedger",
    ),
    Gate(
        "G-EXTERNAL-SHARES",
        (
            "the lift, between mules and matched ordinary receivers, of the external share of new "
            "payers; and between mules and matched ordinary senders, of the external share of "
            "onward P2P value"
        ),
        (
            "`is_external` and the internal or external split of new payers rest on a declared "
            "share of victims at the same bank"
        ),
    ),
    Gate(
        "G-ORIGIN-C",
        (
            'the origin C look-alike lift: the lift of "the customer joined in the 90 days before '
            "t, holds exactly one device link, and its accounts received no payroll credit in the "
            '90 days before t"'
        ),
        '"new and quiet" must not be a mule-only marker',
    ),
    Gate(
        "G-COVER-VOLUME",
        (
            'the cover-volume lift: the lift of "in the 30 days before t the account made more '
            "outgoing P2P, bill and card payments than the 90th percentile of internal deposit "
            'accounts of the same persona and age band on that Monday"'
        ),
        (
            "cover payments stay as behaviour, and their volume alone must not stand in for the "
            "label"
        ),
    ),
)
BY_ENTRY: dict[str, Gate] = {gate.entry: gate for gate in GATES}
# The gates mapped to a feature: each must be in the gate record and pass.
MAPPED = frozenset(gate.entry for gate in GATES if gate.feature)


@dataclass(frozen=True)
class GateRow:
    """A row of a gate record: a gate's entry, its kind and whether it passed."""

    entry: str
    kind: str
    passed: bool


def read_gates(path: Path) -> tuple[GateRow, ...]:
    """The rows of a gate record, once each holds exactly an entry, a kind of KINDS and a pass
    flag, and no entry and kind come twice."""
    found = tomllib.loads(path.read_text())
    if set(found) - {"gate"}:
        raise ValueError(f"{path} holds {sorted(set(found) - {'gate'})} beside its gates")
    items: object = found.get("gate", [])
    if not isinstance(items, list):
        raise ValueError(f"{path} holds its gates as {type(items).__name__}, not a list")
    rows: list[GateRow] = []
    for item in items:
        if (
            not isinstance(item, dict)
            or set(item) != {"entry", "kind", "pass"}
            or not isinstance(item["entry"], str)
            or item["kind"] not in KINDS
            or not isinstance(item["pass"], bool)
        ):
            raise ValueError(f"{path} holds a gate {item!r}, not an entry, a kind and a pass flag")
        rows.append(GateRow(item["entry"], item["kind"], item["pass"]))
    keys = [(row.entry, row.kind) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{path} holds a gate of one kind twice")
    return tuple(rows)


def fingerprint_failures(world: World, record: Mapping[str, object], path: Path) -> list[str]:
    """Why an identity file's fingerprint is not read_world's: its shared fields missing or
    unknown, or the names of those that differ. Names only: the fields hold counts and sums."""
    expected = shared_fields(world)
    try:
        shared = shared_record(record, expected, path)
    except ValueError:
        return ["the identity file lacks fields of the fingerprint, or holds others"]
    differing = [name for name in expected if shared[name] != expected[name]]
    return [f"the fingerprint differs from read_world's on {differing}"] if differing else []


def check_records(world: World, identity_file: Path, gate_file: Path) -> tuple[str, ...]:
    """Why PhantomLedger's records of the world fail what a reported result needs, by name;
    empty when they pass. world is read_world's fields of the loaded world, identity_file the
    identity file data.worlds matched to it, and gate_file its gate record."""
    record = read_record(identity_file)
    failures = fingerprint_failures(world, record, identity_file)
    if record.get(CLEAN_TREE) is not True:
        failures.append("PhantomLedger was not built from a clean tree")
    if record.get(CERTIFIED) is not True:
        failures.append("PhantomLedger's code gates are not certified")
    if not gate_file.is_file():
        return (*failures, "there is no gate record")
    rows = read_gates(gate_file)
    failures += [
        f"the code gate {row.entry} failed" for row in rows if row.kind == CODE and not row.passed
    ]
    failures += [
        f"the world check {row.entry} failed"
        for row in rows
        if row.kind == WORLD and not row.passed and row.entry in MAPPED
    ]
    present = {row.entry for row in rows}
    failures += [f"the gate record lacks {entry}" for entry in sorted(MAPPED - present)]
    return tuple(failures)
