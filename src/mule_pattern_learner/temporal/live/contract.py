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
# The scope visibility phase of each split; unscoped contexts use phase 3.
SPLIT_PHASE = {"train": 1, "validation": 2, "test": 3}
SPLITS = tuple(SPLIT_PHASE)
PHASE_SPLIT = {phase: split for split, phase in SPLIT_PHASE.items()}
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
# The node features the contract fingerprint lists, in its order. Saved models record the
# fingerprint, so the list stays as it is until the server step changes the contract.
CONTRACT_FEATURES = (
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


def fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def contract_fingerprint() -> str:
    """The feature, relation and time-basis contract that checkpoints record.

    The pool groups are left out: TigerGraph never sees them, and only models that
    use them depend on their definitions, which `FeaturePlan.fingerprint` covers.
    """
    return fingerprint(
        {
            "version": CONTRACT_VERSION,
            "basis": BASIS_ID,
            "features": CONTRACT_FEATURES,
            "relations": RELATIONS,
            "rails": RAILS,
            "groups": {k: vars(v) for k, v in FEATURE_GROUPS.items() if k not in POOL_GROUPS},
            "client_groups": sorted(CLIENT_GROUPS - set(POOL_GROUPS)),
            "channels": CHANNELS,
            "amount_ratio_floor": AMOUNT_RATIO_FLOOR,
            "amount_ratio_cap": AMOUNT_RATIO_CAP,
        }
    )


def pool_definition(groups: Sequence[str]) -> dict[str, Any]:
    """What the pool counts of these groups mean, for the input fingerprint."""
    return {
        "version": POOL_ACTIVITY_VERSION,
        "groups": {g: vars(FEATURE_GROUPS[g]) for g in POOL_GROUPS if g in groups},
        "first_inflow_bands": FIRST_INFLOW_BANDS,
        "pass_through": [PASS_THROUGH_SECONDS, *PASS_THROUGH_RATIO],
    }


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
# The pool groups: counts over the payment messages of the context's own candidate pool
# (at most `recent + older + distinct` per relation), not over the account's whole
# history, computed by the client (batching.pool_activity). The numbers are round, but
# the choice of counts followed a diagnostic study that had read the data generator's
# mule typology and test-split mules, so test audits are optimistic for them. The
# internal-payer counts and their amount bands suit the generator, which places scam
# victims inside the bank, more than a real bank, so they are a group of their own
# (pool_internal_inflows) that an ablation can drop.
FIRST_INFLOW_BANDS = (100, 1000)
# Rapid pass-through: the next outflow after an inflow follows within a day and moves
# 50 to 100 percent of the inflow amount.
PASS_THROUGH_SECONDS = 86_400
PASS_THROUGH_RATIO = (0.5, 1.0)
# Changes whenever batching.pool_activity changes what a count means.
POOL_ACTIVITY_VERSION = 1
POOL_ACTIVITY_FEATURES = tuple(
    f"pool_{r}_{v}" for r in RELATIONS[:4] for v in ("count", "unique")
) + ("pool_in_unique", "pool_out_unique", "pool_first_in", "pool_pass_through_1d")
POOL_INTERNAL_FEATURES = ("pool_first_in_internal",) + tuple(
    f"pool_first_in_internal_{band}" for band in FIRST_INFLOW_BANDS
)
POOL_GROUPS = ("pool_activity", "pool_internal_inflows")


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
    # The pool groups are client computed from the context's payment messages and never
    # requested from TigerGraph. First-time counts read the pair_history fields and
    # pass-through counts the flow_timing fields.
    "pool_activity": FeatureGroup(
        "summary", POOL_ACTIVITY_FEATURES, requires=("pair_history", "flow_timing")
    ),
    "pool_internal_inflows": FeatureGroup(
        "summary", POOL_INTERNAL_FEATURES, requires=("pair_history",)
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
CLIENT_GROUPS = frozenset({"hub_indicator", *POOL_GROUPS})
DEFAULT_GROUPS = (
    "entity_meta",
    "hub_indicator",
    "message_core",
    "time_encoding",
    "pair_history",
    "flow_timing",
)
# The groups of the built-in run (config_schema.DEFAULT_RUN): the defaults plus the pool
# groups.
BUILT_IN_GROUPS = (*DEFAULT_GROUPS, *POOL_GROUPS)
# Columns follow registry order. Before the layered restructure a fixed list placed the
# columns of these groups elsewhere, so a plan with one of them fingerprints differently
# now, and a model saved with such a plan is refused instead of misread.
REORDERED_GROUPS = frozenset(
    {"rolling_windows", "amount_ratios", "recency", "association_counts", "pair_window_counts"}
)


@dataclass(frozen=True)
class FeaturePlan:
    """The feature groups a model reads and its architecture.

    "split" is the graph model: attention over sampled neighbours, with a summary
    branch for the root's summary columns. "summary" reads only the root's node and
    summary columns (the controls without attention).
    """

    groups: tuple[str, ...] = BUILT_IN_GROUPS
    architecture: str = "split"

    def __post_init__(self) -> None:
        if len(set(self.groups)) != len(self.groups) or set(self.groups) - FEATURE_GROUPS.keys():
            raise ValueError("Duplicate or unknown feature groups")
        if self.architecture not in ("split", "summary"):
            raise ValueError("Architecture must be split or summary")
        for name in self.groups:
            if set(FEATURE_GROUPS[name].requires) - set(self.groups):
                raise ValueError(f"Missing dependencies for {name}")
        if self.architecture != "summary" and "message_core" not in self.groups:
            raise ValueError("Graph models require message_core")
        if self.architecture == "summary" and not self.names("node", "summary"):
            raise ValueError("Summary model needs node or summary inputs")

    def names(self, *paths: str) -> tuple[str, ...]:
        # Registry order is canonical, independent of configuration list order.
        return tuple(
            n
            for g, spec in FEATURE_GROUPS.items()
            if g in self.groups and spec.path in paths
            for n in spec.names
        )

    @property
    def node_names(self) -> tuple[str, ...]:
        return self.names("node", "summary")

    @property
    def edge_names(self) -> tuple[str, ...]:
        return self.names("message")

    def fingerprint(self) -> str:
        """The model inputs: contract, groups, architecture and any pool definitions."""
        value: dict[str, Any] = {
            "contract": contract_fingerprint(),
            "groups": sorted(self.groups),
            "architecture": self.architecture,
        }
        # The contract leaves the pool groups out, so a plan with one covers them here.
        if set(POOL_GROUPS) & set(self.groups):
            value["pool"] = pool_definition(self.groups)
        if REORDERED_GROUPS & set(self.groups):
            value["columns"] = "registry"
        return fingerprint(value)

    def query_flags(self, hop: int = 1) -> dict[str, bool]:
        """GSQL `include_*` parameters for one hop.

        Channel/stratum are wire metadata even when their embeddings are off, and
        client groups are never requested. Split models read only node and message
        inputs of children, so their second hop skips every summary group. Summary
        models fetch no children.
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
        """feature_groups and architecture, each the built-in run's when absent."""
        groups = tuple(config.get("feature_groups", BUILT_IN_GROUPS))
        return cls(groups, config.get("architecture", "split"))


POOL_KEYS = ("recent", "older", "distinct", "associations", "max_history")
SAMPLER_BACKENDS = ("auto", "cugraph", "torch")
# The keys of a [sampler] table besides the roots pool and [sampler.children].
SAMPLER_KEYS = (
    "relation_fanouts",
    "association_fanout",
    "association_slots",
    "backend",
    "evaluation_seed",
)
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


def _default_children(roots: PoolPlan) -> PoolPlan:
    # The resampled second hop is payments-only, so children skip association candidates.
    return replace(roots, associations=0)


@dataclass(frozen=True, init=False)
class SamplerPlan:
    """Candidate pools per hop plus the client-side neighbour resampling.

    TigerGraph returns each context's candidate pool (`PoolPlan`, one per hop). The
    client draws, per context and relation, at most `relation_fanouts[hop-1]` payment
    candidates (`association_fanout` per association relation at hop 1) uniformly
    without replacement, then merges them into the fanout slots with at most
    `association_slots` of them associations. Hop 2 is payments-only, so the children
    pool defaults to the roots pool without associations.
    """

    roots: PoolPlan
    children: PoolPlan
    relation_fanouts: tuple[int, int]
    association_fanout: int
    association_slots: int
    backend: str
    evaluation_seed: int

    def __init__(
        self,
        *,
        roots: PoolPlan | None = None,
        children: PoolPlan | None = None,
        relation_fanouts: Sequence[int] = (8, 4),
        association_fanout: int = 1,
        association_slots: int = 2,
        backend: str = "auto",
        evaluation_seed: int = 0,
    ) -> None:
        roots = roots if roots is not None else PoolPlan()
        fanouts = tuple(relation_fanouts)
        if len(fanouts) != 2:
            raise ValueError("Sampler relation_fanouts must have one value per hop")
        if backend not in SAMPLER_BACKENDS:
            raise ValueError(f"Sampler backend must be one of {SAMPLER_BACKENDS}")
        values = {
            "roots": roots,
            "children": children if children is not None else _default_children(roots),
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

    def response_bound(self, hop: int = 1) -> int:
        """Maximum messages in one context at this hop."""
        return self.pool(hop).response_bound

    def to_config(self) -> dict[str, Any]:
        """The `[sampler]` table that `from_config` maps back to this plan."""
        values: dict[str, Any] = asdict(self.roots)
        if self.children != _default_children(self.roots):
            values["children"] = asdict(self.children)
        values |= {
            "association_slots": self.association_slots,
            "relation_fanouts": list(self.relation_fanouts),
            "association_fanout": self.association_fanout,
            "backend": self.backend,
            "evaluation_seed": self.evaluation_seed,
        }
        return values

    def fingerprint(self) -> str:
        """Selection semantics for manifest checks; the execution backend is excluded.

        It includes `selection_keys` (SELECTION_KEYS_VERSION), so manifests and
        checkpoints drawn with an older key scheme are not treated as comparable. The
        value keeps the policy name it had while other policies existed, so recorded
        fingerprints still compare equal.
        """
        value = self.to_config()
        value.pop("backend")
        value["children"] = asdict(self.children)
        value["policy"] = "resample"
        value["selection_keys"] = SELECTION_KEYS_VERSION
        return fingerprint(value)

    def pool_fingerprint(self) -> str:
        """Only what TigerGraph is asked for (preparation and cache identity)."""
        return fingerprint({"roots": self.query_params(1), "children": self.query_params(2)})

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> SamplerPlan:
        """Flat `[sampler]` pool keys describe roots; `[sampler.children]` overrides them.

        Configurations saved while other policies existed name this one, `policy =
        "resample"`; any other policy is refused.
        """
        values = dict(config.get("sampler") or {})
        children_values = values.pop("children", None)
        policy = values.pop("policy", "resample")
        if policy != "resample":
            raise ValueError(f"Unknown history sampler {policy!r}: only resample remains")
        unknown = sorted(set(values) - set(POOL_KEYS) - set(SAMPLER_KEYS))
        if unknown:
            raise ValueError(f"Unknown [sampler] key(s): {', '.join(unknown)}")
        roots = PoolPlan(**{k: values.pop(k) for k in POOL_KEYS if k in values})
        children = None
        if children_values is not None:
            if not isinstance(children_values, dict):
                raise ValueError("[sampler.children] must be a table of pool keys")
            unknown = sorted(set(children_values) - set(POOL_KEYS))
            if unknown:
                raise ValueError(f"Unknown [sampler.children] key(s): {', '.join(unknown)}")
            children = replace(_default_children(roots), **children_values)
        return cls(roots=roots, children=children, **values)


def sampler_pools(sampler: SamplerPlan) -> dict[str, dict[str, Any]]:
    """The query-relevant part of a sampler: what TigerGraph returns per hop."""
    return {"roots": sampler.query_params(1), "children": sampler.query_params(2)}


def extraction_groups(config: dict[str, Any]) -> tuple[str, ...]:
    """The configured extraction superset without client groups.

    `extraction_groups`, else `feature_groups`, else the built-in groups.
    """
    groups = config.get("extraction_groups") or config.get("feature_groups") or BUILT_IN_GROUPS
    return tuple(g for g in groups if g not in CLIENT_GROUPS)


def extraction_plan(config: dict[str, Any]) -> FeaturePlan:
    """What the context source asks TigerGraph for.

    Groups are `extraction_groups(config)`; client groups are computed locally.
    The architecture is the model's, so a split model skips summary groups at hop 2.
    """
    model = FeaturePlan.from_config(config)
    groups = extraction_groups(config)
    missing = set(model.groups) - set(groups) - CLIENT_GROUPS
    if missing:
        raise ValueError(
            f"Extraction groups must cover all model inputs; missing {sorted(missing)}"
        )
    return FeaturePlan(groups, model.architecture)
