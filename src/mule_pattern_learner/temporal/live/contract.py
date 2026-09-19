"""Shared feature, relation and query contracts for the live temporal pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from ..encoding import BASIS_ID

CONTRACT_VERSION = "temporal_live_v3_gsql_ratios"
NODE_TYPES = ("Account", "Token", "Party", "Device", "IP", "Address")
WINDOWS = {"1h": 3_600_000, "1d": 86_400_000, "7d": 604_800_000, "30d": 2_592_000_000}
AMOUNT_RATIO_WINDOWS = ("1d", "7d")
AMOUNT_RATIO_FLOOR = 1.0
AMOUNT_RATIO_CAP = 100.0
AMOUNT_RATIO_FEATURES = tuple(f"{window}_out_in_amount_ratio" for window in AMOUNT_RATIO_WINDOWS)
ASSOCIATIONS = (
    ("Party_Owns_Account", "Account_Owned_By_Party"),
    ("Party_Uses_Token", "Token_Used_By_Party"),
    ("Token_Bound_To_Account", "Account_Bound_From_Token"),
    ("Party_Uses_Device", "Device_Used_By_Party"),
    ("Account_Uses_Device", "Device_Used_By_Account"),
    ("Party_Uses_IP", "IP_Used_By_Party"),
    ("Party_Has_Address", "Address_Used_By_Party"),
)
RELATIONS = ("zelle_out", "zelle_in", "payment_out", "payment_in") + tuple(
    name for pair in ASSOCIATIONS for name in pair
)
RAILS = ("unknown", "zelle", "ach", "card", "cash", "check", "internal")
ROLLING_FIELDS = (
    "out_count",
    "in_count",
    "out_amount",
    "in_amount",
    "out_missing",
    "in_missing",
    "out_zelle",
    "in_zelle",
    "out_unique",
    "in_unique",
)
FEATURE_NAMES = (
    tuple("type_" + t for t in NODE_TYPES)
    + ("is_external", "is_deposit", "age_days")
    + tuple(f"{window}_{field}" for window in WINDOWS for field in ROLLING_FIELDS)
    + ("out_recency_days", "in_recency_days", "out_recency_present", "in_recency_present")
    + tuple(f"{r}_{state}" for pair in ASSOCIATIONS for r in pair for state in ("active", "ended"))
    + AMOUNT_RATIO_FEATURES
)


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
        if len(self.node_id.encode()) > 1024 or len(self.scope_id.encode()) > 256:
            raise ValueError("Entity/scope ID exceeds the transport limit")
        if self.visibility_phase not in (1, 2, 3):
            raise ValueError("Visibility phase must be train=1, validation=2 or test=3")
        if self.node_type not in NODE_TYPES or not self.node_id:
            raise ValueError("Unsupported or empty entity")
        if not 0 < self.cutoff_seq < 2**63 or not 0 < self.cutoff_ms < 2**63:
            raise ValueError("Cutoff clocks must be positive signed-64-bit values")


def fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def contract_fingerprint() -> str:
    return fingerprint(
        {
            "version": CONTRACT_VERSION,
            "basis": BASIS_ID,
            "features": FEATURE_NAMES,
            "relations": RELATIONS,
            "rails": RAILS,
            "amount_ratio_floor": AMOUNT_RATIO_FLOOR,
            "amount_ratio_cap": AMOUNT_RATIO_CAP,
        }
    )
