"""The export contract: the 22 tables PhantomLedger's mule use case exports and the loader
loads, as frozen data (docs/design.md, "The export contract").

Each table loads into the graph type of the same name in gsql/schema/schema.gsql. A vertex
table's columns are its type's attributes, `id` its primary id. A link or role table's
`from_id` and `to_id` are its edge's ends, and its other columns are the edge's attributes;
a link's `valid_from_ms` is the edge's discriminator. TigerGraph makes each edge's reverse
type from `reverse`, which has no table of its own. tests/gsql/test_schema.py holds the
schema to these tables, column for column.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Kind(StrEnum):
    """What a table holds."""

    # An entity or a payment.
    VERTEX = "vertex"
    # A tenure from valid_from_ms until valid_to_ms (0 while open): ownership or enrolment.
    LINK = "link"
    # One role of a payment, repeating its event_ts_ms: sender, receiver, token, device, IP.
    ROLE = "role"


@dataclass(frozen=True)
class Column:
    """A column of a table, and the TigerGraph type of the attribute or id it fills."""

    name: str
    graph_type: str


@dataclass(frozen=True)
class Table:
    """An export table and the graph type it loads into, which has its name.

    source and target are an edge's FROM and TO vertex types, and reverse its reverse edge
    type; a vertex table has none of them.
    """

    name: str
    kind: Kind
    columns: tuple[Column, ...]
    source: str | None = None
    target: str | None = None
    reverse: str | None = None

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns)


# A vertex table's first column, its primary id.
ID = Column("id", "STRING")
# An edge table's first two columns: the ids of its FROM and TO vertices.
ENDS = (Column("from_id", "STRING"), Column("to_id", "STRING"))


def _vertex(name: str, *columns: tuple[str, str]) -> Table:
    return Table(name, Kind.VERTEX, (ID, *(Column(*column) for column in columns)))


def _link(name: str, source: str, target: str, reverse: str) -> Table:
    columns = (*ENDS, Column("valid_from_ms", "UINT"), Column("valid_to_ms", "UINT"))
    return Table(name, Kind.LINK, columns, source, target, reverse)


def _role(name: str, source: str, target: str, reverse: str) -> Table:
    return Table(name, Kind.ROLE, (*ENDS, Column("event_ts_ms", "UINT")), source, target, reverse)


# The 22 tables, in the order the schema declares their types.
TABLES: tuple[Table, ...] = (
    _vertex("Party", ("joined_ms", "UINT")),
    _vertex(
        "Account",
        ("account_type", "STRING"),
        ("is_external", "BOOL"),
        ("opened_ms", "UINT"),
        ("closed_ms", "UINT"),
        ("is_mule", "INT"),
        ("pu_label", "INT"),
        ("mule_ring_id", "INT"),
        ("mule_onset_ms", "UINT"),
        ("mule_last_ms", "UINT"),
    ),
    _vertex("Token"),
    _vertex("Device"),
    _vertex("IP"),
    _vertex(
        "Payment_Transaction",
        ("amount", "DOUBLE"),
        ("payment_rail", "STRING"),
        ("event_ts_ms", "UINT"),
    ),
    _vertex("Zelle_Transfer", ("amount", "DOUBLE"), ("event_ts_ms", "UINT")),
    _link("Party_Owns_Account", "Party", "Account", "Account_Owned_By_Party"),
    _link("Party_Uses_Token", "Party", "Token", "Token_Used_By_Party"),
    _link("Token_Bound_To_Account", "Token", "Account", "Account_Bound_From_Token"),
    _link("Party_Uses_Device", "Party", "Device", "Device_Used_By_Party"),
    _link("Party_Uses_IP", "Party", "IP", "IP_Used_By_Party"),
    _role(
        "Transaction_From_Account",
        "Payment_Transaction",
        "Account",
        "Account_Initiated_Transaction",
    ),
    _role(
        "Transaction_To_Account", "Payment_Transaction", "Account", "Account_Received_Transaction"
    ),
    _role("Transaction_Used_Device", "Payment_Transaction", "Device", "Device_Used_By_Transaction"),
    _role("Transaction_Used_IP", "Payment_Transaction", "IP", "IP_Observed_In_Transaction"),
    _role("Transfer_From_Account", "Zelle_Transfer", "Account", "Account_Sent_Zelle_Transfer"),
    _role("Transfer_To_Account", "Zelle_Transfer", "Account", "Account_Received_Zelle_Transfer"),
    _role("Transfer_From_Token", "Zelle_Transfer", "Token", "Token_Sent_Transfer"),
    _role("Transfer_To_Token", "Zelle_Transfer", "Token", "Token_Received_Transfer"),
    _role("Transfer_Used_Device", "Zelle_Transfer", "Device", "Device_Used_By_Transfer"),
    _role("Transfer_Used_IP", "Zelle_Transfer", "IP", "IP_Used_By_Transfer"),
)

# The closed value lists of account_type and payment_rail. mule prepare refuses a world
# holding any other value; a new one is a contract change in the design, PhantomLedger and
# the loader together.
ACCOUNT_TYPES = ("deposit", "credit", "brokerage", "gl", "unknown")
PAYMENT_RAILS = ("ach", "card", "cash", "check", "internal", "unknown")

# The Account columns that carry a label or the truth, which only read_labels and
# read_answer_key read (docs/design.md, "Who may read what").
LABEL_COLUMNS = ("is_mule", "pu_label", "mule_ring_id", "mule_onset_ms", "mule_last_ms")

# The label convention of is_mule and pu_label: 1, 0 for any other internal account, and
# EXTERNAL_LABEL exactly on external accounts.
CONVENTION_COLUMNS = ("is_mule", "pu_label")
LABEL_VALUES = (-1, 0, 1)
EXTERNAL_LABEL = -1


def follows_label_convention(value: int, *, is_external: bool) -> bool:
    """Whether an account's is_mule or pu_label value keeps the convention: -1 when the
    account is external, and 0 or 1 when it is not."""
    if type(value) is not int or value not in LABEL_VALUES:
        return False
    return (value == EXTERNAL_LABEL) == is_external
