"""The graph as the queries see it: node types, relations, splits and context keys.

Nothing here reads the graph. The values name vertex and edge types, the rails and
channels of payment messages, the sampling strata, the split phases of the frozen
scope, the columns of a hub row and the Account columns of the loading contract.
"""

from __future__ import annotations

from dataclasses import dataclass

from .bounds import ID_BYTES, SCOPE_ID_BYTES

NODE_TYPES = ("Account", "Token", "Party", "Device", "IP", "Address")
ASSOCIATIONS = (
    ("Party_Owns_Account", "Account_Owned_By_Party"),
    ("Party_Uses_Token", "Token_Used_By_Party"),
    ("Token_Bound_To_Account", "Account_Bound_From_Token"),
    ("Party_Uses_Device", "Device_Used_By_Party"),
    ("Account_Uses_Device", "Device_Used_By_Account"),
    ("Party_Uses_IP", "IP_Used_By_Party"),
    ("Party_Has_Address", "Address_Used_By_Party"),
)
# Target types of each (forward, reverse) association pair, aligned with ASSOCIATIONS.
ASSOCIATION_TARGETS = (
    ("Account", "Party"),
    ("Token", "Party"),
    ("Account", "Token"),
    ("Device", "Party"),
    ("Device", "Account"),
    ("IP", "Party"),
    ("Address", "Party"),
)
# The relations whose neighbours are payment messages; they come first in RELATIONS.
PAYMENT_RELATIONS = ("zelle_out", "zelle_in", "payment_out", "payment_in")
ASSOCIATION_RELATIONS = tuple(name for pair in ASSOCIATIONS for name in pair)
RELATIONS = PAYMENT_RELATIONS + ASSOCIATION_RELATIONS
# The position of each relation in RELATIONS: its code in candidate tables and batches.
RELATION_INDEX = {name: i for i, name in enumerate(RELATIONS)}
RAILS = ("unknown", "zelle", "ach", "card", "cash", "check", "internal")
# The scope visibility phase of each split; unscoped contexts use phase 3.
SPLIT_PHASE = {"train": 1, "validation": 2, "test": 3}
SPLITS = tuple(SPLIT_PHASE)
# The splits held out from training: the proxy predicts them and the audits cover them.
HELD_OUT_SPLITS: tuple[str, ...] = ("validation", "test")
PHASE_SPLIT = {phase: split for split, phase in SPLIT_PHASE.items()}


@dataclass(frozen=True, order=True)
class ContextKey:
    """History is seq < cutoff_seq AND timestamp <= cutoff_ms.

    Association visibility uses seed_seq=cutoff_seq-1. A calendar cutoff is
    converted to the preceding millisecond before its sequence is resolved.
    """

    node_type: str
    node_id: str
    cutoff_seq: int
    cutoff_ms: int
    scope_id: str = ""
    visibility_phase: int = 3

    def __post_init__(self) -> None:
        if len(self.node_id.encode()) > ID_BYTES or len(self.scope_id.encode()) > SCOPE_ID_BYTES:
            raise ValueError("Entity/scope ID exceeds the transport limit")
        if self.visibility_phase not in PHASE_SPLIT:
            raise ValueError("Visibility phase must be train=1, validation=2 or test=3")
        if self.node_type not in NODE_TYPES or not self.node_id:
            raise ValueError("Unsupported or empty entity")
        if not 0 < self.cutoff_seq < 2**63 or not 0 < self.cutoff_ms < 2**63:
            raise ValueError("Cutoff clocks must be positive signed-64-bit values")

    @property
    def batch_phase(self) -> int:
        """The phase of a batch of this root: its own when scoped, 3 when unscoped."""
        return self.visibility_phase if self.scope_id else 3


# Unknown categorical values have a dedicated bucket, never an observed category.
# The loaded data only carries digital, branch_or_atm, bank and unknown, 1:1 with rail.
CHANNELS = (
    "unknown",
    "digital",
    "branch_or_atm",
    "bank",
    "p2p",
    "atm_withdrawal",
    "card_purchase",
    "online",
    "mobile",
    "branch",
    "other",
)
STRATA = ("recent", "older", "distinct", "association")
# Every run keeps its splits in disjoint scope partitions (strict inductive); outputs
# record it so their performance claims say what they cover.
EVALUATION_PROTOCOL = "strict_inductive"


def context_scope(scope_id: object) -> str:
    """The scope id of every context of a dataset; strict inductive splits need one."""
    if not isinstance(scope_id, str) or not scope_id:
        raise ValueError("Strict inductive sampling requires a frozen TigerGraph scope id")
    return scope_id


# The columns of a hub row, in order, as the hub query prints them and the prepared hub
# registry stores them, each with its type. A hub is listed for one reason only.
HUB_COLUMNS: dict[str, type[int] | type[str]] = {
    "account_id": str,
    "cutoff_seq": int,
    "visibility_phase": int,
    "max_visible": int,
    "max_degree": int,
    "reason": str,
}
HUB_REASONS = ("visible_history",)

# The columns of the oracle's truth table (tigergraph.oracle, evaluation.truth), read from
# the ground-truth query. is_mule is 1 or 0, and -1 where the label is not known; ring_id
# is the account's ring of mules, -1 for none; label_source says where the label came
# from (the reveal records its version, its salt and whether the mule was revealed).
TRUTH_COLUMNS = ("account_id", "is_mule", "ring_id", "label_source")


# Account CSV/PSV input columns (gsql/schema/account_loading.gsql): five
# account facts, then the ten supervision fields of the label contract.
ACCOUNT_LOAD_COLUMNS = [
    "id",
    "account_type",
    "is_external",
    "first_seen_seq",
    "first_seen_ts_ms",
    "is_mule",
    "mule_label_known",
    "is_mule_masked",
    "pu_label",
    "mule_label_effective_seq",
    "mule_label_effective_ts_ms",
    "mule_label_available_seq",
    "mule_label_available_ts_ms",
    "mule_ring_id",
    "mule_label_source",
]
