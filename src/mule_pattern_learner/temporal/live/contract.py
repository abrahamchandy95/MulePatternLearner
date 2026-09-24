"""Shared feature, relation and query contracts for the live temporal pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import operator
from typing import Any

from ..encoding import BASIS_ID

CONTRACT_VERSION = "temporal_live_v5_candidate_pools"
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
            "groups": {k: vars(v) for k, v in FEATURE_GROUPS.items()},
            "client_groups": sorted(CLIENT_GROUPS),
            "channels": CHANNELS,
            "amount_ratio_floor": AMOUNT_RATIO_FLOOR,
            "amount_ratio_cap": AMOUNT_RATIO_CAP,
        }
    )


# Unknown categorical values have a dedicated bucket, never an observed category.
# Live data only carries digital, branch_or_atm, bank and unknown, 1:1 with rail.
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
HALF_LIVES = {"1d": 86_400_000, "7d": 604_800_000, "30d": 2_592_000_000, "90d": 7_776_000_000}


@dataclass(frozen=True)
class FeatureGroup:
    path: str
    names: tuple[str, ...]
    identity: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()


FEATURE_GROUPS = {
    "entity_meta": FeatureGroup(
        "node",
        tuple("type_" + t for t in NODE_TYPES) + ("is_external", "is_deposit"),
        tuple("type_" + t for t in NODE_TYPES) + ("is_external", "is_deposit"),
    ),
    "entity_age": FeatureGroup("node", ("age_days",)),
    # Client computed from the hub registry; never requested from TigerGraph.
    "hub_indicator": FeatureGroup("node", ("history_withheld",), identity=("history_withheld",)),
    "history_support": FeatureGroup(
        "summary", ("visible_event_count", "history_lt_5_events"), ("history_lt_5_events",)
    ),
    "rolling_windows": FeatureGroup(
        "summary", tuple(f"{w}_{f}" for w in WINDOWS for f in ROLLING_FIELDS)
    ),
    "amount_ratios": FeatureGroup("summary", AMOUNT_RATIO_FEATURES, requires=("rolling_windows",)),
    "recency": FeatureGroup(
        "summary",
        ("out_recency_days", "in_recency_days", "out_recency_present", "in_recency_present"),
        # Keep the legacy log1p transform for these two baseline flags.
    ),
    "association_counts": FeatureGroup(
        "summary",
        tuple(
            f"{r}_{state}" for pair in ASSOCIATIONS for r in pair for state in ("active", "ended")
        ),
    ),
    "decayed_activity": FeatureGroup(
        "summary",
        tuple(
            f"decay_{h}_{d}_{v}"
            for h in HALF_LIVES
            for d in ("out", "in")
            for v in ("count", "amount")
        ),
    ),
    "identity_order": FeatureGroup(
        "summary",
        tuple(
            f"{r}_{state}_last10"
            for r in ("Account_Owned_By_Party", "Account_Bound_From_Token", "Account_Uses_Device")
            for state in ("starts", "ends")
        ),
    ),
    "message_core": FeatureGroup(
        "message", ("amount", "amount_present", "is_event"), ("amount_present", "is_event")
    ),
    "time_encoding": FeatureGroup(
        "message",
        ("gap_present",)
        + tuple(f"age_fourier_{i}" for i in range(64))
        + tuple(f"gap_fourier_{i}" for i in range(64)),
    ),
    "pair_window_counts": FeatureGroup(
        "message", ("pair_count_1h", "pair_count_1d", "pair_count_7d")
    ),
    "pair_history": FeatureGroup(
        "message",
        ("pair_prior_count", "pair_first_age_seconds", "pair_first_present"),
        ("pair_first_present",),
    ),
    "flow_timing": FeatureGroup(
        "message",
        (
            "flow_delay_seconds",
            "flow_present",
            "flow_censored",
            "flow_observation_seconds",
            "flow_amount_ratio",
            "flow_ratio_present",
            "flow_same_rail",
        ),
        ("flow_present", "flow_censored", "flow_ratio_present", "flow_same_rail"),
    ),
    "device_ip_context": FeatureGroup(
        "message",
        ("device_age_seconds", "device_present", "ip_age_seconds", "ip_present"),
        ("device_present", "ip_present"),
    ),
    "event_channel": FeatureGroup("categorical", ("channel",)),
    "sampler_meta": FeatureGroup("categorical", ("stratum",)),
}
LEGACY_GROUPS = (
    "entity_meta",
    "entity_age",
    "rolling_windows",
    "recency",
    "association_counts",
    "amount_ratios",
    "message_core",
    "time_encoding",
    "pair_window_counts",
)
CLIENT_GROUPS = frozenset({"hub_indicator"})
DEFAULT_GROUPS = (
    "entity_meta",
    "hub_indicator",
    "message_core",
    "time_encoding",
    "pair_history",
    "flow_timing",
)


@dataclass(frozen=True)
class FeaturePlan:
    groups: tuple[str, ...] = LEGACY_GROUPS
    architecture: str = "single"

    def __post_init__(self) -> None:
        if len(set(self.groups)) != len(self.groups) or set(self.groups) - FEATURE_GROUPS.keys():
            raise ValueError("Duplicate or unknown feature groups")
        if self.architecture not in ("single", "split", "summary"):
            raise ValueError("Architecture must be single, split or summary")
        for name in self.groups:
            if set(FEATURE_GROUPS[name].requires) - set(self.groups):
                raise ValueError(f"Missing dependencies for {name}")
        if self.architecture != "summary" and "message_core" not in self.groups:
            raise ValueError("Graph models require message_core")
        if self.architecture == "summary" and not self.names("node", "summary"):
            raise ValueError("Summary model needs node or summary inputs")

    def names(self, *paths: str) -> tuple[str, ...]:
        # Registry order is canonical, independent of configuration list order.
        names = tuple(
            n
            for g, spec in FEATURE_GROUPS.items()
            if g in self.groups and spec.path in paths
            for n in spec.names
        )
        return tuple(n for n in FEATURE_NAMES if n in names) + tuple(
            n for n in names if n not in FEATURE_NAMES
        )

    @property
    def node_names(self) -> tuple[str, ...]:
        return self.names("node", "summary")

    @property
    def edge_names(self) -> tuple[str, ...]:
        # Preserve the shipped baseline ordering, including its Fourier offsets.
        core = FEATURE_GROUPS["message_core"].names if "message_core" in self.groups else ()
        time = ("gap_present",) if "time_encoding" in self.groups else ()
        windows = (
            FEATURE_GROUPS["pair_window_counts"].names
            if "pair_window_counts" in self.groups
            else ()
        )
        fourier = (
            FEATURE_GROUPS["time_encoding"].names[1:] if "time_encoding" in self.groups else ()
        )
        extra = tuple(
            n
            for g in ("pair_history", "flow_timing", "device_ip_context")
            if g in self.groups
            for n in FEATURE_GROUPS[g].names
        )
        return core + time + windows + fourier + extra

    def fingerprint(self) -> str:
        return fingerprint(
            {
                "contract": contract_fingerprint(),
                "groups": sorted(self.groups),
                "architecture": self.architecture,
            }
        )

    def query_flags(self, hop: int = 1) -> dict[str, bool]:
        """GSQL `include_*` parameters for one hop.

        Channel/stratum are wire metadata even when their embeddings are off, and
        client groups are never requested. Split models read only node and message
        inputs of children, so their second hop skips every summary group.
        """
        if hop not in (1, 2):
            raise ValueError("Hop must be 1 or 2")
        skip_summary = hop == 2 and self.architecture == "split"
        return {
            "include_" + name: name in self.groups and not (skip_summary and spec.path == "summary")
            for name, spec in FEATURE_GROUPS.items()
            if spec.path != "categorical" and name != "message_core" and name not in CLIENT_GROUPS
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> FeaturePlan:
        groups = tuple(config.get("feature_groups", LEGACY_GROUPS))
        variant = config.get("variant", "temporal")
        if variant == "no_fourier":
            groups = tuple(g for g in groups if g != "time_encoding")
        architecture = "summary" if variant == "tabular" else config.get("architecture", "single")
        return cls(groups, architecture)


POOL_KEYS = ("recent", "older", "distinct", "associations", "max_history")
SAMPLER_POLICIES = ("recent", "stratified", "resample")
SAMPLER_BACKENDS = ("auto", "cugraph", "torch")
# Defaults of the fields that only the resample policy reads.
RESAMPLE_DEFAULTS = {
    "relation_fanouts": (8, 4),
    "association_fanout": 1,
    "backend": "auto",
    "evaluation_seed": 0,
}
SAMPLER_KEYS = ("policy", "association_slots", *RESAMPLE_DEFAULTS)
# Version of the resample key scheme (sampler.selection_keys), part of the fingerprint.
# 2: evaluation keys mix the hop in (hop 1 unchanged, hop 2 an independent stream).
SELECTION_KEYS_VERSION = 2


def _bounded(owner: str, name: str, value: object, low: int, high: int) -> int:
    try:
        number = operator.index(value)  # type: ignore[arg-type]
    except TypeError:
        number = None
    if isinstance(value, bool) or number is None or not low <= number <= high:
        raise ValueError(f"{owner} {name} must be an integer in [{low},{high}], got {value!r}")
    return number


@dataclass(frozen=True)
class PoolPlan:
    """Bounded, cutoff-safe candidate pool that TigerGraph returns per context and hop.

    Per payment relation: the `recent` most recent visible events, `older` rank
    quantiles and `distinct` events with new peers. Per association relation:
    `associations` valid-time associations. A context whose visible history in one
    relation exceeds `max_history` is rejected (history_capacity_exceeded).
    """

    recent: int = 2
    older: int = 0
    distinct: int = 0
    associations: int = 2
    max_history: int = 2048

    def __post_init__(self) -> None:
        for name, low, high in (
            ("recent", 1, 32),
            ("older", 0, 16),
            ("distinct", 0, 16),
            ("associations", 0, 8),
            ("max_history", 32, 4096),
        ):
            object.__setattr__(self, name, _bounded("Pool", name, getattr(self, name), low, high))

    @property
    def response_bound(self) -> int:
        """Maximum messages in one context: 4 payment and 14 association relations."""
        return 4 * (self.recent + self.older + self.distinct) + 14 * self.associations

    def query_params(self) -> dict[str, int]:
        return {
            "per_relation": self.recent,
            "k_old": self.older,
            "k_div": self.distinct,
            "k_assoc": self.associations,
            "max_history": self.max_history,
        }


class HopBound(int):
    """The hop-1 response bound; call it as `bound(hop)` for another hop."""

    _plan: SamplerPlan

    def __new__(cls, plan: SamplerPlan) -> HopBound:
        bound = super().__new__(cls, plan.roots.response_bound)
        bound._plan = plan
        return bound

    def __call__(self, hop: int = 1) -> int:
        return self._plan.pool(hop).response_bound

    def __reduce__(self) -> tuple[type[int], tuple[int]]:
        # Pickles, copies and checkpoints store the plain hop-1 integer.
        return int, (int(self),)


def _default_children(policy: str, roots: PoolPlan) -> PoolPlan:
    # The resampled second hop is payments-only, so children skip association candidates.
    return replace(roots, associations=0) if policy == "resample" else roots


@dataclass(frozen=True, init=False)
class SamplerPlan:
    """Candidate pools per hop plus the client-side neighbor selection policy.

    `recent` and `stratified` are the deterministic legacy selections. `resample`
    draws, per context and relation, at most `relation_fanouts[hop-1]` payment
    candidates (`association_fanout` per association relation at hop 1) uniformly
    without replacement, then merges them into the fanout slots. Hop 2 is
    payments-only. The legacy forms `SamplerPlan("stratified", 4, 3, 2, 2, 2048)`
    and `SamplerPlan(recent=4)` still describe the roots pool; children then
    default to the roots pool (without associations under `resample`).
    """

    policy: str
    roots: PoolPlan
    children: PoolPlan
    relation_fanouts: tuple[int, int]
    association_fanout: int
    association_slots: int
    backend: str
    evaluation_seed: int

    def __init__(
        self,
        policy: str = "recent",
        *legacy: int,
        roots: PoolPlan | None = None,
        children: PoolPlan | None = None,
        relation_fanouts: Sequence[int] = (8, 4),
        association_fanout: int = 1,
        association_slots: int = 2,
        backend: str = "auto",
        evaluation_seed: int = 0,
        **pool: int,
    ) -> None:
        if len(legacy) > len(POOL_KEYS):
            raise TypeError("SamplerPlan takes at most five legacy pool values")
        pool = dict(zip(POOL_KEYS, legacy)) | pool
        if set(pool) - set(POOL_KEYS):
            raise TypeError(f"Unexpected sampler arguments: {sorted(set(pool) - set(POOL_KEYS))}")
        if pool and roots is not None:
            raise TypeError("Pass either roots=PoolPlan(...) or legacy pool values, not both")
        roots = roots if roots is not None else PoolPlan(**pool)
        if policy not in SAMPLER_POLICIES:
            raise ValueError(f"Unknown history sampler {policy!r}")
        children = children if children is not None else _default_children(policy, roots)
        # Legacy callers are untyped; keep the runtime check.
        if not isinstance(roots, PoolPlan) or not isinstance(children, PoolPlan):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError("Sampler pools must be PoolPlan values")
        if policy == "recent" and any(p.older or p.distinct for p in (roots, children)):
            raise ValueError("Recent sampler cannot request older/distinct strata")
        fanouts = tuple(relation_fanouts)
        if len(fanouts) != 2:
            raise ValueError("Sampler relation_fanouts must have one value per hop")
        values = {
            "policy": policy,
            "roots": roots,
            "children": children,
            "relation_fanouts": tuple(
                _bounded("Sampler", "relation_fanouts", v, 1, 64) for v in fanouts
            ),
            "association_fanout": _bounded(
                "Sampler", "association_fanout", association_fanout, 0, 8
            ),
            "association_slots": _bounded("Sampler", "association_slots", association_slots, 0, 16),
            "backend": backend,
            "evaluation_seed": _bounded(
                "Sampler", "evaluation_seed", evaluation_seed, 0, 2**63 - 1
            ),
        }
        if backend not in SAMPLER_BACKENDS:
            raise ValueError(f"Sampler backend must be one of {SAMPLER_BACKENDS}")
        if policy != "resample":
            changed = [k for k, v in RESAMPLE_DEFAULTS.items() if values[k] != v]
            if policy == "recent" and values["association_slots"] != 2:
                changed.append("association_slots")
            if changed:
                raise ValueError(f"{', '.join(changed)} do not apply to the {policy} sampler")
        for name, value in values.items():
            object.__setattr__(self, name, value)

    def pool(self, hop: int) -> PoolPlan:
        if hop == 1:
            return self.roots
        if hop == 2:
            return self.children
        raise ValueError("Hop must be 1 or 2")

    def query_params(self, hop: int = 1) -> dict[str, int]:
        return self.pool(hop).query_params()

    @property
    def response_bound(self) -> HopBound:
        """Hop-1 bound as an int; `response_bound(hop)` gives any hop."""
        return HopBound(self)

    # Legacy flat attributes describe the roots pool.
    @property
    def recent(self) -> int:
        return self.roots.recent

    @property
    def older(self) -> int:
        return self.roots.older

    @property
    def distinct(self) -> int:
        return self.roots.distinct

    @property
    def associations(self) -> int:
        return self.roots.associations

    @property
    def max_history(self) -> int:
        return self.roots.max_history

    def to_config(self) -> dict[str, Any]:
        """The `[sampler]` table that `from_config` maps back to this plan."""
        values: dict[str, Any] = {"policy": self.policy, **asdict(self.roots)}
        if self.children != _default_children(self.policy, self.roots):
            values["children"] = asdict(self.children)
        if self.policy != "recent":
            values["association_slots"] = self.association_slots
        if self.policy == "resample":
            values |= {k: getattr(self, k) for k in RESAMPLE_DEFAULTS}
            values["relation_fanouts"] = list(self.relation_fanouts)
        return values

    def fingerprint(self) -> str:
        """Selection semantics for manifest checks; the execution backend is excluded.

        Resample plans include `selection_keys` (SELECTION_KEYS_VERSION), so manifests
        and checkpoints drawn with an older key scheme are not treated as comparable.
        """
        value = self.to_config()
        value.pop("backend", None)
        value["children"] = asdict(self.children)
        if self.policy == "resample":
            value["selection_keys"] = SELECTION_KEYS_VERSION
        return fingerprint(value)

    def pool_fingerprint(self) -> str:
        """Only what TigerGraph is asked for (preparation and cache identity)."""
        return fingerprint({"roots": self.query_params(1), "children": self.query_params(2)})

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> SamplerPlan:
        """Flat `[sampler]` pool keys describe roots; `[sampler.children]` overrides them."""
        values = dict(config.get("sampler") or {})
        children_values = values.pop("children", None)
        unknown = sorted(set(values) - set(POOL_KEYS) - set(SAMPLER_KEYS))
        if unknown:
            raise ValueError(f"Unknown [sampler] key(s): {', '.join(unknown)}")
        pool = {"recent": int(config.get("per_relation", 2))}
        pool |= {k: values.pop(k) for k in POOL_KEYS if k in values}
        roots = PoolPlan(**pool)
        policy = values.pop("policy", "recent")
        children = None
        if children_values is not None:
            if not isinstance(children_values, dict):
                raise ValueError("[sampler.children] must be a table of pool keys")
            unknown = sorted(set(children_values) - set(POOL_KEYS))
            if unknown:
                raise ValueError(f"Unknown [sampler.children] key(s): {', '.join(unknown)}")
            children = replace(_default_children(policy, roots), **children_values)
        return cls(policy, roots=roots, children=children, **values)
